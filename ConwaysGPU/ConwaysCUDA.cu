#include "ConwaysCUDA.h"
#include "pattern_blueprints.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"  // From NVIDIA CUDA samples.

#include <cmath>
#include <iostream>
#include <algorithm>

using ConwaysCUDA::CELL_AGE_T;
using ConwaysCUDA::CELL_STATUS_T;

namespace {

	enum PatternMode: uint8_t {
		NORMAL, HOVER, UNHOVER	
	};

	// A utility class for sending Patterns to the GPU.
	class PatternWrapper {
	public:
		~PatternWrapper()
		{
			checkCudaErrors(cudaFree(d_pattern));
			checkCudaErrors(cudaFree(d_pattern_str));
		}

		void init(int max_pattern_size)
		{
			checkCudaErrors(cudaMalloc(&d_pattern, sizeof(Pattern)));
			checkCudaErrors(cudaMalloc(&d_pattern_str, sizeof(char) * max_pattern_size));
		}
		
		void to_gpu(const Pattern& pattern)
		{
			size_t pattern_bytes = pattern.rows * pattern.cols * sizeof(char);
			cudaMemcpy(d_pattern_str, pattern.pattern, pattern_bytes, cudaMemcpyHostToDevice);

			Pattern pattern_cpy = pattern;
			pattern_cpy.pattern = d_pattern_str;
			cudaMemcpy(d_pattern, &pattern_cpy, sizeof(Pattern), cudaMemcpyHostToDevice);
		}

		const Pattern* get_pattern_ptr() const { return d_pattern; }
	private:
		Pattern* d_pattern;
		char* d_pattern_str;
	};

	// A utility class for mapping OpenGL VBOs for CUDA usage.
	class WorldVBOMapper {
	public:
		WorldVBOMapper(cudaGraphicsResource* resources[ConwaysCUDA::_VBO_COUNT])
			: resources(resources)
		{
			checkCudaErrors(cudaGraphicsMapResources(ConwaysCUDA::_VBO_COUNT, resources, 0));

			size_t num_bytes;
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_cell_status_ptr, &num_bytes, resources[ConwaysCUDA::CELL_STATUS]));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_cell_age_ptr, &num_bytes, resources[ConwaysCUDA::CELL_AGE]));
		}

		~WorldVBOMapper() 
		{
			checkCudaErrors(cudaGraphicsUnmapResources(ConwaysCUDA::_VBO_COUNT, resources, 0));
		}

		CELL_STATUS_T* d_cell_status_ptr;
		CELL_AGE_T* d_cell_age_ptr;
	private:
		cudaGraphicsResource** resources;
	};

	CELL_AGE_T* d_prev_cell_age_data;  // (in GPU memory).
	// Unfortunately, using __managed__ would mean having to cudaDeviceSynchronize
	// each time before use (to support older GPUs) ... That is why I'm using 
	// seperate host and device variables.
	__device__ int d_world_rows, d_world_cols;
	int world_rows, world_cols;

	cudaGraphicsResource* vbo_resources[ConwaysCUDA::_VBO_COUNT];

	const size_t MAX_PATTERN_SIZE = 1<<10;  // scratch pad memory for PatternWrapper
	PatternWrapper pattern_wrapper;

	// Device constants.
	// Trying to initialize these two arrays in the kernel resulted 
	// in illegal memory address problems.
	__constant__ int DX[8] = { 1, 1, 1, 0, 0, -1, -1, -1 };
	__constant__ int DY[8] = { 0, 1, -1, 1, -1, 0, 1, -1 };
}


// Forward declaration.
void set_pattern(const Blueprint & blueprint, int row, int col, PatternMode type);


__device__
inline bool in_range(int n, int lo, int hi)
{
	return n >= lo && n <= hi;
}

// The following two functions define a bijective mapping 
// between normal and hovered tile state.
__device__
CELL_STATUS_T cell_to_hovered(CELL_STATUS_T n, bool pattern_bit)
{
	if (!in_range(n, -8, 1)) return n;
	if (pattern_bit) return n + 30;			// [-8, 1] -> [22, 31]
	return n + 10;							// [-8, 1] -> [2, 11]
}

__device__
CELL_STATUS_T hovered_to_cell(CELL_STATUS_T n)
{
	if (in_range(n, 22, 31)) return n - 30;
	if (in_range(n, 2, 11)) return n - 10;
	return n;
}

__device__
inline bool is_cell_hovered(CELL_STATUS_T n)
{
	return !in_range(n, -8, 1);
}

__device__
inline bool is_cell_hovered_bit(CELL_STATUS_T n)
{
	return n >= 22;
}

__global__
void copy_mem(const CELL_AGE_T* src, CELL_AGE_T* dst, size_t size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int i = index; i < size; i += stride)
		dst[i] = src[i];
}

__device__
inline int get_world_index(int row, int col)
{
	// % (modulo) is used to wrap around the grid to give an illusion
	// of an infinite grid. 
	// -1 % 5 -> -1  [c++]
	// -1 % 5 ->  4  [python]
	// Solution for negative numbers when doing a mod n:
	// a + n mod n, because a + k*n mod n == a mod n !
	col = (d_world_cols + col) % d_world_cols;
	row = (d_world_rows + row) % d_world_rows;
	return row * d_world_cols + col;
}

// First component is the row, the second is the col.
__device__
inline int2 get_grid_cell(int world_idx, int cols)
{
	return make_int2(world_idx / cols, world_idx % cols);
}

__device__
int count_neighbours(CELL_AGE_T* old_state, int row, int col)
{
	int neighbours = 0;
	for (int i = 0; i < 8; ++i) {
		int neighbour_idx = get_world_index(row + DY[i], col + DX[i]);
		if (old_state[neighbour_idx] > 0)
			++neighbours;
	}
	return neighbours;
}

__global__
void tick_kernel(CELL_AGE_T* old_cell_ages, CELL_STATUS_T* new_cell_status, CELL_AGE_T* new_cell_ages)
{
	// 1. Any live cell with fewer than two live neighbors dies, as if by underpopulation.
	// 2. Any live cell with two or three live neighbors lives on to the next generation.
	// 3. Any live cell with more than three live neighbors dies, as if by overpopulation.
	// 4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

	// A grid-stride loop.
	// Each thread in the grid has its own threadIdx, which means
	// we can calculate for each thread which elements of an 
	// array it should process without interfering with any other
	// thread. 
	// index calculation: number of blocks in grid * index of current block + thread index.
	// stride tells us the total amount of threads.
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int size = d_world_rows * d_world_cols;
	for (int i = index; i < size; i += stride) {
		int2 tile = get_grid_cell(i, d_world_cols);
		int neighbours = count_neighbours(old_cell_ages, tile.x, tile.y);

		bool cell_alive = old_cell_ages[i] > 0;
		if (neighbours == 3 || (cell_alive && neighbours == 2)) {
			// If a pattern is being hovered at this location,
			// we have to keep it hovered and figure out if the pattern bit was set.
			int cell_status = new_cell_status[i];
			if (is_cell_hovered(cell_status)) {
				new_cell_status[i] = cell_to_hovered(1, is_cell_hovered_bit(cell_status));
			} else {
				new_cell_status[i] = 1;
			}
			new_cell_ages[i] = max(1, new_cell_ages[i] + 1);
		}
		else {
			int cell_status = new_cell_status[i];
			if (is_cell_hovered(cell_status)) {
				new_cell_status[i] = cell_to_hovered(-neighbours, is_cell_hovered_bit(cell_status));
			} else {
				new_cell_status[i] = -neighbours;
			}
			new_cell_ages[i] = min(-1, new_cell_ages[i] - 1);
		}
	}
}

// Simulates one iteration of the game of life.
void ConwaysCUDA::tick()
{
	// 0. map the VBO for use by CUDA.
	// 1. copy the current world state as defined by the VBO.
	// 2. update the current state in the VBO.
	// 3. unmap the VBO.
	// Steps 0 and 3 are done by WorldVBOMapper

	WorldVBOMapper mapped_vbos(vbo_resources);

	// CUDA threads execute in a _grid_ of of threads. Each block in the grid is
	// of size block_size, and num_blocks is how many blocks there are in the grid.
	// Total number of simultaneous threads is then block_size * num_blocks.
	// When running a kernel function (__global__) (GPU code), we specify the size
	// of the thread grid.
	const size_t grid_size = world_rows * world_cols;
	const size_t block_size = 256;
	int num_blocks = std::ceil(grid_size / block_size);

	// Copy the previous world state using cudaMemcpy or copy_mem kernel:
	//cudaMemcpy(world_state, data_ptr, grid_size * sizeof(GLbyte), cudaMemcpyDeviceToDevice);
	copy_mem<<<num_blocks, block_size>>>(mapped_vbos.d_cell_age_ptr, 
										 d_prev_cell_age_data, grid_size);

	tick_kernel<<<num_blocks, block_size>>>(d_prev_cell_age_data, 
											mapped_vbos.d_cell_status_ptr, 
											mapped_vbos.d_cell_age_ptr);
}

__global__
void toggle_cell_kernel(CELL_AGE_T* cell_age, CELL_STATUS_T* cell_status, int idx)
{
	CELL_STATUS_T status = cell_status[idx];
	cell_age[idx] = (status > 0 ? 0 : 1);
	cell_status[idx] = cell_age[idx];
}

// There are 3 options for pattern building:
//   1. normal pattern
//   2. hover
//   3. remove hover
__global__
void build_pattern_kernel(CELL_AGE_T* cell_age, CELL_STATUS_T* cell_status, 
						  const Pattern* pattern, int row, int col, PatternMode type)
{
	// Thread index of cell in pattern.
	int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int size = pattern->cols * pattern->rows;
	for (int cell_idx = thread_idx; cell_idx < size; cell_idx += stride) {
		int2 pattern_cell = get_grid_cell(cell_idx, pattern->cols);
		char val = pattern->pattern[cell_idx] - '0';

		// (0,0) is at the bottom left
		int idx = get_world_index(row - pattern_cell.x, col + pattern_cell.y);
		switch (type) {
		case NORMAL: 
			cell_age[idx] = val;
			cell_status[idx] = val; 
			break;
		case HOVER: 
			cell_status[idx] = cell_to_hovered(cell_status[idx], val); 
			break;
		case UNHOVER: 
			cell_status[idx] = hovered_to_cell(cell_status[idx]); 
			break;
		}
	}
}

void set_pattern(const Pattern& pattern, int row, int col, PatternMode type)
{
	// This function is the biggest CPU bottleneck when hovering patterns!
	// It would be more efficient to batch all the patterns and process them
	// in a kernel. Another option would be to send all the patterns to
	// the GPU only once at program start-up. 

	WorldVBOMapper mapped_vbos(vbo_resources);

	// We have to send the Pattern to the GPU.
	pattern_wrapper.to_gpu(pattern);

	// Make a thread for each cell in the pattern.
	const size_t grid_size = pattern.width() * pattern.height();
	const size_t block_size = 64;
	int num_blocks = std::ceil(grid_size / (double)block_size);
	build_pattern_kernel<<<num_blocks, block_size>>>(mapped_vbos.d_cell_age_ptr, 
		mapped_vbos.d_cell_status_ptr, pattern_wrapper.get_pattern_ptr(), row, col, type);
}

void set_pattern(const MultiPattern& pattern, int row, int col, PatternMode type)
{
	for (int i = 0; i < pattern.blueprints.size(); ++i) {
		set_pattern(*pattern.blueprints[i], 
					row + pattern.row_offsets[i], 
					col + pattern.col_offsets[i], type);
	}
}

void set_pattern(const Blueprint & blueprint, int row, int col, PatternMode type)
{
	// TODO: a better way to do this? 
	switch (blueprint.type()) {
	case Blueprint::BlueprintType::Pattern: 
		set_pattern(static_cast<const Pattern&>(blueprint), row, col, type);
		break;
	case Blueprint::BlueprintType::MultiPattern:
		set_pattern(static_cast<const MultiPattern&>(blueprint), row, col, type);
		break;
	}
}

void ConwaysCUDA::set_pattern(const Blueprint & blueprint, int row, int col)
{
	set_pattern(blueprint, row, col, NORMAL);
}

void ConwaysCUDA::set_hover_pattern(const Blueprint & blueprint, int row, int col, bool hover)
{
	// Hovered state information is written into the cell_status vbo.
	// That information is then used in the fragment shader to highlight
	// the given pattern. By using cell_to_hovered and hovered_to_cell,
	// we make sure no cell_status information is lost. The cell_age vbo
	// is left untouched when hovering patterns.
	set_pattern(blueprint, row, col, (hover ? HOVER : UNHOVER));
}

void ConwaysCUDA::toggle_cell(int row, int col)
{
	WorldVBOMapper mapped_vbos(vbo_resources);
	// The alternative to using glMapBuffer, etc...
	toggle_cell_kernel<<<1, 1>>>(mapped_vbos.d_cell_age_ptr, 
								 mapped_vbos.d_cell_status_ptr, 
								 row * world_cols + col);
}

bool ConwaysCUDA::init(int rows, int cols, GLuint vbos[_VBO_COUNT])
{
	world_rows = rows;
	world_cols = cols;

	cudaMemcpyToSymbol(d_world_rows, &world_rows, sizeof(int));
	cudaMemcpyToSymbol(d_world_cols, &world_cols, sizeof(int));

	checkCudaErrors(cudaMalloc(&d_prev_cell_age_data, sizeof(CELL_AGE_T) * rows * cols));

	pattern_wrapper.init(MAX_PATTERN_SIZE);

	// Necessary for OpenGL interop.
	for (int i = 0; i < _VBO_COUNT; ++i)
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(
			&vbo_resources[i], vbos[i], cudaGraphicsRegisterFlagsNone));

	return true;
}

void ConwaysCUDA::exit()
{
	for (int i = 0; i < _VBO_COUNT; ++i)
		checkCudaErrors(cudaGraphicsUnregisterResource(vbo_resources[i]));
	checkCudaErrors(cudaFree(d_prev_cell_age_data));
}
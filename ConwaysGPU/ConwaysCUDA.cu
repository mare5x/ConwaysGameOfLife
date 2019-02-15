#include "ConwaysCUDA.h"
#include "pattern_blueprints.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"  // From NVIDIA CUDA samples.

#include <cmath>
#include <iostream>
#include <algorithm>
#include <unordered_map>

using ConwaysCUDA::CELL_AGE_T;
using ConwaysCUDA::CELL_STATUS_T;

namespace {
	typedef unsigned int uint;

	enum PatternMode: uint8_t {
		NORMAL, HOVER, UNHOVER	
	};

	// A utility class for sending Patterns to the GPU (with caching).
	class PatternCache {
	public:
		~PatternCache()
		{
			for (const auto& entry : cache)
				checkCudaErrors(cudaFree(entry.second));
		}
		
		const char* to_gpu(const Pattern& pattern)
		{
			// Check if it's cached.
			auto entry = cache.find(pattern.pattern);
			if (entry != cache.end())
				return entry->second;

			// Copy only the pattern string to the GPU.
			char* d_pattern_str;
			size_t pattern_bytes = pattern.rows * pattern.cols * sizeof(char);

			checkCudaErrors(cudaMalloc(&d_pattern_str, pattern_bytes));
				
			checkCudaErrors(
				cudaMemcpy(d_pattern_str, pattern.pattern, 
						   pattern_bytes, cudaMemcpyHostToDevice)
			);

			// Save it to the cache.
			auto pair = cache.insert({ pattern.pattern, d_pattern_str });
			return pair.first->second;
		}
	private:
		// Cache from <pattern string> to <device pattern string>.
		std::unordered_map<const char*, char*> cache;
	};

	// A utility class for mapping OpenGL VBOs for CUDA usage.
	class WorldVBOMapper {
	public:
		void init(GLuint vbos[ConwaysCUDA::_VBO_COUNT])
		{
			for (int i = 0; i < ConwaysCUDA::_VBO_COUNT; ++i)
				checkCudaErrors(cudaGraphicsGLRegisterBuffer(
					&vbo_resources[i], vbos[i], cudaGraphicsRegisterFlagsNone));
		}

		void exit()
		{
			for (int i = 0; i < ConwaysCUDA::_VBO_COUNT; ++i)
				checkCudaErrors(cudaGraphicsUnregisterResource(vbo_resources[i]));
		}

		void map()
		{
			if (_is_mapped) return;

			checkCudaErrors(
				cudaGraphicsMapResources(ConwaysCUDA::_VBO_COUNT, vbo_resources, 0)
			);

			size_t num_bytes;
			checkCudaErrors(
				cudaGraphicsResourceGetMappedPointer(
					(void **)&d_cell_status_ptr, 
					&num_bytes, 
					vbo_resources[ConwaysCUDA::CELL_STATUS])
			);
			checkCudaErrors(
				cudaGraphicsResourceGetMappedPointer(
					(void **)&d_cell_age_ptr, 
					&num_bytes, 
					vbo_resources[ConwaysCUDA::CELL_AGE])
			);

			_is_mapped = true;
		}

		void unmap()
		{
			checkCudaErrors(
				cudaGraphicsUnmapResources(ConwaysCUDA::_VBO_COUNT, vbo_resources, 0)
			);
			_is_mapped = false;
		}

		//bool is_mapped() const { return _is_mapped; }

		CELL_STATUS_T* get_cell_status() { return d_cell_status_ptr; }
		CELL_AGE_T* get_cell_age() { return d_cell_age_ptr; }
	private:
		bool _is_mapped = false;

		cudaGraphicsResource* vbo_resources[ConwaysCUDA::_VBO_COUNT];

		CELL_STATUS_T* d_cell_status_ptr;
		CELL_AGE_T* d_cell_age_ptr;
	};

	CELL_AGE_T* d_prev_cell_age_data;  // (in GPU memory).
	// Unfortunately, using __managed__ would mean having to cudaDeviceSynchronize
	// each time before use (to support older GPUs) ... That is why I'm using 
	// seperate host and device variables.
	__device__ int d_world_rows, d_world_cols;
	int world_rows, world_cols;

	WorldVBOMapper vbo_mapper;
	PatternCache pattern_cache;

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

	// CUDA threads execute in a _grid_ of of threads. Each block in the grid is
	// of size block_size, and num_blocks is how many blocks there are in the grid.
	// Total number of simultaneous threads is then block_size * num_blocks.
	// When running a kernel function (__global__) (GPU code), we specify the size
	// of the thread grid.
	const uint block_size = 256;
	uint grid_size = world_rows * world_cols;
	uint num_blocks = (grid_size + block_size - 1) / block_size;

	// Copy the previous world state using cudaMemcpy or copy_mem kernel:
	//cudaMemcpy(world_state, data_ptr, grid_size * sizeof(GLbyte), cudaMemcpyDeviceToDevice);
	copy_mem<<<num_blocks, block_size>>>(vbo_mapper.get_cell_age(), 
										 d_prev_cell_age_data, grid_size);

	tick_kernel<<<num_blocks, block_size>>>(d_prev_cell_age_data, 
											vbo_mapper.get_cell_status(), 
											vbo_mapper.get_cell_age());
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
						  const Pattern pattern, int row, int col, PatternMode type)
{
	// Thread index of cell in pattern.
	int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int size = pattern.cols * pattern.rows;
	for (int cell_idx = thread_idx; cell_idx < size; cell_idx += stride) {
		int2 pattern_cell = get_grid_cell(cell_idx, pattern.cols);
		char val = pattern.pattern[cell_idx] - '0';

		pattern.get_rotated_coordinates(&pattern_cell.x, &pattern_cell.y);

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
	// The option I chose was the simplest: caching in PatternCache.
	// That fixed the to_gpu() problem. But now the bottleneck is in WorldVBOMapper.
	// The solution is to map right at the start of pattern building 
	// (in set_pattern(Blueprint ...)) or some earlier time ...

	// We have to send the Pattern to the GPU. The pattern string
	// gets cached and the pattern contents get copied to the GPU.
	// However, we have to point pattern.pattern to the device string
	// otherwise the GPU would try to access host memory.
	const char* d_pattern_str = pattern_cache.to_gpu(pattern);
	Pattern pattern_cpy = pattern;
	pattern_cpy.pattern = d_pattern_str;

	// Make a thread for each cell in the pattern.
	const uint block_size = 64;
	uint grid_size = pattern.width() * pattern.height();
	uint num_blocks = (grid_size + block_size - 1) / block_size;
	build_pattern_kernel<<<num_blocks, block_size>>>(
		vbo_mapper.get_cell_age(), vbo_mapper.get_cell_status(), 
		pattern_cpy, row, col, type);
}

void set_pattern(const MultiPattern& pattern, int row, int col, PatternMode type)
{
	for (int i = 0; i < pattern.blueprints.size(); ++i) {
		int row_offset = pattern.row_offsets[i];
		int col_offset = pattern.col_offsets[i];
		pattern.get_rotated_offset(pattern.blueprints[i], row_offset, col_offset);
		set_pattern(*pattern.blueprints[i], 
					row + row_offset, 
					col + col_offset, type);
	}
}

void set_pattern(const Blueprint & blueprint, int row, int col, PatternMode type)
{
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
	// The alternative to using glMapBuffer, etc...
	toggle_cell_kernel<<<1, 1>>>(vbo_mapper.get_cell_age(), 
								 vbo_mapper.get_cell_status(), 
								 row * world_cols + col);
}

bool ConwaysCUDA::init(int rows, int cols, GLuint vbos[_VBO_COUNT])
{
	world_rows = rows;
	world_cols = cols;

	cudaMemcpyToSymbol(d_world_rows, &world_rows, sizeof(int));
	cudaMemcpyToSymbol(d_world_cols, &world_cols, sizeof(int));

	checkCudaErrors(cudaMalloc(&d_prev_cell_age_data, sizeof(CELL_AGE_T) * rows * cols));

	// Necessary for OpenGL interop.
	vbo_mapper.init(vbos);

	return true;
}

void ConwaysCUDA::exit()
{
	vbo_mapper.exit();
	checkCudaErrors(cudaFree(d_prev_cell_age_data));
}

void ConwaysCUDA::start_interop()
{
	vbo_mapper.map();
}

void ConwaysCUDA::stop_interop()
{
	vbo_mapper.unmap();
}

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

	// A utility class for sending Patterns to the GPU.
	class PatternWrapper {
	public:
		~PatternWrapper()
		{
			checkCudaErrors(cudaFree(dev_pattern));
			checkCudaErrors(cudaFree(dev_pattern_str));
		}

		void init(int max_pattern_size)
		{
			checkCudaErrors(cudaMallocManaged(&dev_pattern, sizeof(Pattern)));
			checkCudaErrors(cudaMallocManaged(&dev_pattern_str, sizeof(char) * max_pattern_size));
		}
		
		void to_gpu(const Pattern& pattern)
		{
			// Remove this line to get an exception :)
			cudaDeviceSynchronize();
			
			strcpy(dev_pattern_str, pattern.pattern);

			*dev_pattern = pattern;
			dev_pattern->pattern = dev_pattern_str;
		}

		const Pattern* get_pattern_ptr() const { return dev_pattern; }
	private:
		Pattern* dev_pattern;
		char* dev_pattern_str;
	};

	CELL_AGE_T* prev_cell_age_data;  // (in GPU memory).
	int rows, cols;
	cudaGraphicsResource* vbo_resources[ConwaysCUDA::_VBO_COUNT];

	const size_t MAX_PATTERN_SIZE = 1<<10;  // scratch pad memory for PatternWrapper
	PatternWrapper pattern_wrapper;

	// Device constants.
	// Trying to initialize these two arrays in the kernel resulted 
	// in illegal memory address problems.
	__constant__ int DX[8] = { 1, 1, 1, 0, 0, -1, -1, -1 };
	__constant__ int DY[8] = { 0, 1, -1, 1, -1, 0, 1, -1 };
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
int count_neighbours(CELL_AGE_T* old_state, int row, int col, int rows, int cols)
{
	int neighbours = 0;
	for (int i = 0; i < 8; ++i) {
		// % (modulo) is used to wrap around the grid to give an illusion
		// of an infinite grid. 
		// -1 % 5 -> -1  [c++]
		// -1 % 5 ->  4  [python]
		// Solution for negative numbers when doing a mod n:
		// a + n mod n, because a + k*n mod n == a mod n !
		int neighbour_col = (cols + (col + DX[i])) % cols;
		int neighbour_row = (rows + (row + DY[i])) % rows;
		int neighbour_idx = neighbour_row * cols + neighbour_col;
		if (old_state[neighbour_idx] > 0)
			++neighbours;
	}
	return neighbours;
}

__global__
void tick_kernel(CELL_AGE_T* old_cell_ages, CELL_STATUS_T* new_cell_status, CELL_AGE_T* new_cell_ages, int rows, int cols)
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

	int size = rows * cols;
	for (int i = index; i < size; i += stride) {
		int row = i / cols;
		int col = i % cols;

		int neighbours = count_neighbours(old_cell_ages, row, col, rows, cols);

		bool cell_alive = old_cell_ages[i] > 0;
		if (neighbours == 3 || (cell_alive && neighbours == 2)) {
			new_cell_status[i] = 1;
			new_cell_ages[i] = max(1, new_cell_ages[i] + 1);
		}
		else {
			new_cell_status[i] = -neighbours;
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

	checkCudaErrors(cudaGraphicsMapResources(_VBO_COUNT, vbo_resources, 0));

	CELL_STATUS_T* cell_status_ptr;
	CELL_AGE_T* cell_age_ptr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cell_status_ptr, &num_bytes, vbo_resources[CELL_STATUS]));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cell_age_ptr, &num_bytes, vbo_resources[CELL_AGE]));

	// CUDA threads execute in a _grid_ of of threads. Each block in the grid is
	// of size block_size, and num_blocks is how many blocks there are in the grid.
	// Total number of simultaneous threads is then block_size * num_blocks.
	// When running a kernel function (__global__) (GPU code), we specify the size
	// of the thread grid.
	const size_t grid_size = rows * cols;
	const size_t block_size = 256;
	int num_blocks = std::ceil(grid_size / block_size);

	// Copy the previous world state using cudaMemcpy or copy_mem kernel:
	//cudaMemcpy(world_state, data_ptr, grid_size * sizeof(GLbyte), cudaMemcpyDeviceToDevice);
	copy_mem<<<num_blocks, block_size>>>(cell_age_ptr, prev_cell_age_data, grid_size);

	tick_kernel<<<num_blocks, block_size>>>(prev_cell_age_data, cell_status_ptr, cell_age_ptr, rows, cols);

	checkCudaErrors(cudaGraphicsUnmapResources(_VBO_COUNT, vbo_resources, 0));
}

__global__
void toggle_cell_kernel(CELL_AGE_T* cell_age, CELL_STATUS_T* cell_status, int idx)
{
	CELL_STATUS_T status = cell_status[idx];
	cell_age[idx] = (status > 0 ? 0 : 1);
	cell_status[idx] = cell_age[idx];
}

__global__
void build_pattern_kernel(CELL_AGE_T* cell_age, CELL_STATUS_T* cell_status, const Pattern* pattern, int rows, int cols, int row, int col)
{
	// Thread index of cell in pattern.
	int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int size = pattern->cols * pattern->rows;

	for (int cell_idx = thread_idx; cell_idx < size; cell_idx += stride) {
		int pattern_row = cell_idx / pattern->cols;
		int pattern_col = cell_idx % pattern->cols;

		char val = pattern->pattern[cell_idx] - '0';

		// (0,0) is at the bottom left
		int world_row = (rows + row - pattern_row) % rows;
		int world_col = (cols + col + pattern_col) % cols;
		int idx = world_row * cols + world_col;
		cell_age[idx] = val;
		cell_status[idx] = val;
	}
}

void set_pattern(const Pattern& pattern, int row, int col)
{
	using namespace ConwaysCUDA;

	checkCudaErrors(cudaGraphicsMapResources(_VBO_COUNT, vbo_resources, 0));

	CELL_STATUS_T* cell_status_ptr;
	CELL_AGE_T* cell_age_ptr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cell_status_ptr, &num_bytes, vbo_resources[CELL_STATUS]));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cell_age_ptr, &num_bytes, vbo_resources[CELL_AGE]));

	// We have to send the Pattern to the GPU.
	pattern_wrapper.to_gpu(pattern);

	// Make a thread for each cell in the pattern.
	const size_t grid_size = pattern.width() * pattern.height();
	const size_t block_size = 64;
	int num_blocks = std::ceil(grid_size / (double)block_size);
	build_pattern_kernel<<<num_blocks, block_size>>>(cell_age_ptr, cell_status_ptr, pattern_wrapper.get_pattern_ptr(), rows, cols, row, col);

	checkCudaErrors(cudaGraphicsUnmapResources(_VBO_COUNT, vbo_resources, 0));
}

void set_pattern(const MultiPattern& pattern, int row, int col)
{
	const auto& blueprints = pattern.blueprints;
	const auto& row_offsets = pattern.row_offsets;
	const auto& col_offsets = pattern.col_offsets;
	for (int i = 0; i < blueprints.size(); ++i) {
		const Blueprint& blueprint = *blueprints[i];
		ConwaysCUDA::set_pattern(blueprint, row + row_offsets[i], col + col_offsets[i]);
	}
}

void ConwaysCUDA::set_pattern(const Blueprint & blueprint, int row, int col)
{
	// TODO: a better way to do this? 
	switch (blueprint.type()) {
	case Blueprint::BlueprintType::Pattern: 
		set_pattern(static_cast<const Pattern&>(blueprint), row, col);
		break;
	case Blueprint::BlueprintType::MultiPattern:
		set_pattern(static_cast<const MultiPattern&>(blueprint), row, col);
		break;
	}
}

void ConwaysCUDA::toggle_cell(int row, int col)
{
	checkCudaErrors(cudaGraphicsMapResources(_VBO_COUNT, vbo_resources, 0));

	CELL_STATUS_T* cell_status_ptr;
	CELL_AGE_T* cell_age_ptr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cell_status_ptr, &num_bytes, vbo_resources[CELL_STATUS]));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cell_age_ptr, &num_bytes, vbo_resources[CELL_AGE]));

	// The alternative to using glMapBuffer, etc...
	toggle_cell_kernel<<<1, 1>>>(cell_age_ptr, cell_status_ptr, row * cols + col);

	checkCudaErrors(cudaGraphicsUnmapResources(_VBO_COUNT, vbo_resources, 0));
}

bool ConwaysCUDA::init(int rows, int cols, GLuint vbos[_VBO_COUNT])
{
	::rows = rows;
	::cols = cols;
	checkCudaErrors(cudaMallocManaged(&prev_cell_age_data, sizeof(CELL_AGE_T) * rows * cols));

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
	checkCudaErrors(cudaFree(prev_cell_age_data));
}
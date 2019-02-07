#include "ConwaysCUDA.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"  // From NVIDIA CUDA samples.

#include <cmath>
#include <iostream>
#include <algorithm>

using ConwaysCUDA::CELL_AGE_T;
using ConwaysCUDA::CELL_STATUS_T;

namespace {
	CELL_AGE_T* prev_cell_age_data;  // (in GPU memory).
	int rows, cols;
	cudaGraphicsResource* vbo_resources[ConwaysCUDA::_VBO_COUNT];

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

void ConwaysCUDA::toggle_cell(int row, int col)
{
	checkCudaErrors(cudaGraphicsMapResources(_VBO_COUNT, vbo_resources, 0));

	CELL_STATUS_T* cell_status_ptr;
	CELL_AGE_T* cell_age_ptr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cell_status_ptr, &num_bytes, vbo_resources[CELL_STATUS]));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&cell_age_ptr, &num_bytes, vbo_resources[CELL_AGE]));

	// The alternative to 
	toggle_cell_kernel<<<1, 1>>>(cell_age_ptr, cell_status_ptr, row * cols + col);

	checkCudaErrors(cudaGraphicsUnmapResources(_VBO_COUNT, vbo_resources, 0));
}

bool ConwaysCUDA::init(int rows, int cols, GLuint vbos[_VBO_COUNT])
{
	::rows = rows;
	::cols = cols;
	checkCudaErrors(cudaMallocManaged(&prev_cell_age_data, sizeof(CELL_AGE_T) * rows * cols));

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
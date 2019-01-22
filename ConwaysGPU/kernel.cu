#include "ConwaysCUDA.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"  // From NVIDIA CUDA samples.

#include <cmath>
#include <iostream>


namespace {
	float* world_state;  // Represents the "previous" state of the world (in GPU memory).
	int rows, cols;
	cudaGraphicsResource* vbo_resource;

	// Device constants.
	// Trying to initialize these two arrays in the kernel resulted 
	// in illegal memory address problems.
	__constant__ int DX[8] = { 1, 1, 1, 0, 0, -1, -1, -1 };
	__constant__ int DY[8] = { 0, 1, -1, 1, -1, 0, 1, -1 };
}


__global__
void copy_mem(const float* src, float* dst, size_t size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;
	for (int i = index; i < size; i += stride)
		dst[i] = src[i];
}

__device__
int count_neighbours(float* old_state, int row, int col, int rows, int cols)
{
	int neighbours = 0;
	for (int i = 0; i < 8; ++i) {
		// % (modulo) is used to wrap around the grid to give an illusion
		// of an infinite grid. 
		int neighbour_col = (col + DX[i]) % cols;
		int neighbour_row = (row + DY[i]) % rows;
		int neighbour_idx = neighbour_row * cols + neighbour_col;
		if (old_state[neighbour_idx] > 0.01f)
			++neighbours;
	}
	return neighbours;
}

__global__
void run_kernel(float* old_state, float* world, int rows, int cols)
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

		int neighbours = count_neighbours(old_state, row, col, rows, cols);

		bool cell_alive = old_state[i] > 0.5f;
		if (neighbours == 3 || (cell_alive && neighbours == 2))
			world[i] = 1.0f;
		else
			world[i] = 0.0f;
	}
}

// Simulates one iteration of the game of life.
void ConwaysCUDA::tick()
{
	// 0. map the VBO for use by CUDA.
	// 1. copy the current world state as defined by the VBO.
	// 2. update the current state in the VBO.
	// 3. unmap the VBO.

	checkCudaErrors(cudaGraphicsMapResources(1, &vbo_resource, 0));

	float* data_ptr;
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&data_ptr, &num_bytes, vbo_resource));

	// CUDA threads execute in a _grid_ of of threads. Each block in the grid is
	// of size block_size, and num_blocks is how many blocks there are in the grid.
	// Total number of simultaneous threads is then block_size * num_blocks.
	// When running a kernel function (__global__) (GPU code), we specify the size
	// of the thread grid.
	const size_t grid_size = rows * cols;
	const size_t block_size = 256;
	int num_blocks = std::ceil(grid_size / block_size);

	// Copy the previous world state using cudaMemcpy or copy_mem kernel:
	//cudaMemcpy(world_state, data_ptr, grid_size * sizeof(float), cudaMemcpyDeviceToDevice);
	copy_mem<<<num_blocks, block_size>>>(data_ptr, world_state, grid_size);

	run_kernel<<<num_blocks, block_size>>>(world_state, data_ptr, rows, cols);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &vbo_resource, 0));
}

bool ConwaysCUDA::init(int rows, int cols, GLuint vbo)
{
	::rows = rows;
	::cols = cols;
	checkCudaErrors(cudaMallocManaged(&world_state, sizeof(float) * rows * cols));

	// Necessary for OpenGL interop.
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&vbo_resource, vbo, cudaGraphicsRegisterFlagsNone));

	return true;
}

void ConwaysCUDA::exit()
{
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_resource));
	checkCudaErrors(cudaFree(world_state));
}
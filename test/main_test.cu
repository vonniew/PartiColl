#include <gtest/gtest.h>
#include <algorithm>
#include <type_traits>

constexpr int blocksize = 16;

__global__ void fillStuff(int *in) {
	in[threadIdx.x] = threadIdx.x;
}

__global__ void runStuff(int *in, int *out) {
	out[threadIdx.x] = in[threadIdx.x] * 2;
}

TEST(MainCudaTests, CudaWorks) {
	dim3 block{blocksize, 1};
	dim3 grid{1, 1};


	int *in, *out;

	cudaMalloc((void**)&in, blocksize * sizeof(int));
	cudaMalloc((void**)&out, blocksize * sizeof(int));

	fillStuff<<<grid, block>>>(in);
	runStuff<<<grid, block>>>(in, out);

	int result[blocksize];

	cudaMemcpy(result, out, blocksize, cudaMemcpyHostToDevice);

	cudaFree(in);
	cudaFree(out);
}
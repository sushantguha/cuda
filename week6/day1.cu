#include <iostream>

__device__ __noinline__ int work_a(int v) {
    for (int i = 0; i < 200; i++) v = v * 3 + i;
    return v;
}

__device__ __noinline__ int work_b(int v) {
    for (int i = 0; i < 800; i++) v = v * 5 + i;
    return v;
}

__global__ void branch(int* out) {
    int v;
    if ((threadIdx.x % 2) == 0) {
        v = work_a(0);
    } else {
        v = work_b(1);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = v;
}

int main() {
    const int blocks  = 1024;
    const int threads = 256;
    const int N       = blocks * threads;

    int* d_out;
    cudaMalloc(&d_out, N * sizeof(int));

    branch<<<blocks, threads>>>(d_out);
    cudaDeviceSynchronize();

    cudaFree(d_out);
    return 0;
}

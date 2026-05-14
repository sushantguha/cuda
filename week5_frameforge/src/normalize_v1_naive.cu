#include <stdio.h>
#include <cstdint>

__global__ void normalize_v1_naive(const uint8_t* __restrict__ in,
                                   float*          __restrict__ out,
                                   int   N,
                                   float mu_r,  float mu_g,  float mu_b,
                                   float inv_sigma_r, float inv_sigma_g, float inv_sigma_b)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("%d %d %d %d %d %d %d\n", gridDim.x, blockDim.x, blockIdx.x, threadIdx.x, tid, warpSize, tid);
    }
    int mod = tid % 3;
    if (mod == 0) {
        out[tid] = (in[tid] - mu_r) * inv_sigma_r;
    } else if (mod == 1) {
        out[tid] = (in[tid] - mu_g) * inv_sigma_g;
    } else {
        out[tid] = (in[tid] - mu_b) * inv_sigma_b;
    }
}
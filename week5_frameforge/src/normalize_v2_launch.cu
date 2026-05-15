#include <cstdio>
#include <cstdint>

__global__ void normalize_v2_launch(const uint8_t* __restrict__ in,
                                   float*          __restrict__ out,
                                   int   N,
                                   float mu_r,  float mu_g,  float mu_b,
                                   float inv_sigma_r, float inv_sigma_g, float inv_sigma_b)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("[gpu] launch: gridDim.x=%d blockDim.x=%d warpSize=%d N=%d\n",
               gridDim.x, blockDim.x, warpSize, N);
        printf("[gpu] params: mu=(%.4f, %.4f, %.4f) inv_sigma=(%.4f, %.4f, %.4f)\n",
               mu_r, mu_g, mu_b, inv_sigma_r, inv_sigma_g, inv_sigma_b);
    }

    int   mod = tid % 3;
    float result;
    if      (mod == 0) result = (in[tid] - mu_r) * inv_sigma_r;
    else if (mod == 1) result = (in[tid] - mu_g) * inv_sigma_g;
    else               result = (in[tid] - mu_b) * inv_sigma_b;
    out[tid] = result;

    if (blockIdx.x == 0 && threadIdx.x < 3) {
        const char* ch = (threadIdx.x == 0) ? "R" : (threadIdx.x == 1) ? "G" : "B";
        printf("[gpu] tid=%d ch=%s in=%u out=%.4f\n",
               tid, ch, (unsigned) in[tid], result);
    }
}

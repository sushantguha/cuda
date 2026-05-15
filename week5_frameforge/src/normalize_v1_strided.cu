#include <cstdint>

// Anti-coalesced reference kernel: consecutive threads in a warp read addresses
// gridDim.x elements apart instead of 1 apart.
// Indexing: tid = threadIdx.x * gridDim.x + blockIdx.x
// Compare against normalize_v2_launch (tid = blockIdx.x * blockDim.x + threadIdx.x).
__global__ void normalize_v1_strided(const uint8_t* __restrict__ in,
                                     float*          __restrict__ out,
                                     int   N,
                                     float mu_r,  float mu_g,  float mu_b,
                                     float inv_sigma_r, float inv_sigma_g, float inv_sigma_b)
{
    int tid = threadIdx.x * gridDim.x + blockIdx.x;
    if (tid >= N) return;

    int   mod = tid % 3;
    float result;
    if      (mod == 0) result = (in[tid] - mu_r) * inv_sigma_r;
    else if (mod == 1) result = (in[tid] - mu_g) * inv_sigma_g;
    else               result = (in[tid] - mu_b) * inv_sigma_b;
    out[tid] = result;
}

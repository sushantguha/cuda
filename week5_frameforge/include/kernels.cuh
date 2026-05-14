#pragma once
#include <cstdint>

// ---- Day 1 ----
__global__ void normalize_v1_naive(const uint8_t* __restrict__ in,
                                   float*         __restrict__ out,
                                   int N,
                                   float mu_r,        float mu_g,        float mu_b,
                                   float inv_sigma_r, float inv_sigma_g, float inv_sigma_b);

// ---- Day 2+ kernels declared here as they're added ----
// __global__ void normalize_v2_launch(...);
// __global__ void normalize_v3_coalesced(...);
// __global__ void pipeline_v4_streams(...);
// __global__ void pipeline_v5_apod(...);

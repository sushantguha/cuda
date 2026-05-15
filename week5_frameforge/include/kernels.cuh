#pragma once
#include <cstdint>

__global__ void normalize_v1_naive(const uint8_t* __restrict__ in,
                                   float*         __restrict__ out,
                                   int N,
                                   float mu_r,        float mu_g,        float mu_b,
                                   float inv_sigma_r, float inv_sigma_g, float inv_sigma_b);

__global__ void normalize_v2_launch(const uint8_t* __restrict__ in,
                                   float*          __restrict__ out,
                                   int   N,
                                   float mu_r,  float mu_g,  float mu_b,
                                   float inv_sigma_r, float inv_sigma_g, float inv_sigma_b);

__global__ void normalize_v1_strided(const uint8_t* __restrict__ in,
                                     float*          __restrict__ out,
                                     int   N,
                                     float mu_r,  float mu_g,  float mu_b,
                                     float inv_sigma_r, float inv_sigma_g, float inv_sigma_b);

__global__ void normalize_v3_coalesced(const uint8_t* __restrict__ in,
                                   float*          __restrict__ out,
                                   int   N,
                                   float mu_r,  float mu_g,  float mu_b,
                                   float inv_sigma_r, float inv_sigma_g, float inv_sigma_b);
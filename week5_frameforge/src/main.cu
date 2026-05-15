#include "frame.hpp"
#include "kernels.cuh"
#include <cstdio>
#include <iostream>
#include "bench.hpp"
#include <vector>
#include <algorithm>

#define WIDTH 1280
#define HEIGHT 960
#define CHANNELS 3
#define PIXELS WIDTH * HEIGHT * CHANNELS

#define LOG(...)                                          \
  do {                                                    \
    std::fprintf(stderr, "[host] " __VA_ARGS__);          \
    std::fprintf(stderr, "\n");                           \
  } while (0)

constexpr float mu[3]        = { 0.485f, 0.456f, 0.406f };
constexpr float inv_sigma[3] = { 1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f };

int main() {
    const int N    = PIXELS;
    

    // LOG("FrameForge v1_naive  W=%d H=%d C=%d  N=%d", WIDTH, HEIGHT, CHANNELS, N);
    // LOG("launch config        block=%d grid=%d threads=%d", BLOCK_SIZE, grid, BLOCK_SIZE * grid);
    // LOG("mu        = (%.4f, %.4f, %.4f)", mu[0], mu[1], mu[2]);
    // LOG("inv_sigma = (%.4f, %.4f, %.4f)", inv_sigma[0], inv_sigma[1], inv_sigma[2]);

    FrameBuffer<uint8_t> uint8Val(N);
    FrameBuffer<float>   fpVal(N);
    uint8_t* h_data   = new uint8_t[N];
    float*   out      = new float[N];
    float*   expected = new float[N];
    std::vector<int> medianTimes = {};
    std::vector<int> blockSizes = {32, 64, 128, 256, 512, 1024};

    for (int i = 0; i < N; i++) h_data[i] = i % 256;
    // LOG("input ready — h_data[0..5] = %u %u %u %u %u %u", h_data[0], h_data[1], h_data[2], h_data[3], h_data[4], h_data[5]);

    uint8Val.copy_from_host(h_data);
    // LOG("H2D done — %zu bytes copied", (size_t) N * sizeof(uint8_t));

    // LOG("launching normalize_v1_naive<<<%d, %d>>>", grid, BLOCK_SIZE);

    // Sweep through block sizes
    for (int BLOCK_SIZE : blockSizes) {
        const int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        std::vector<int> times = {};

        for (int i = 0; i < 51; i++) {
            
            Timer timer;
            if (i) {
                timer.tic();
            }

            normalize_v2_launch<<<grid, BLOCK_SIZE>>>(
                uint8Val.dataPtr, fpVal.dataPtr, N,
                mu[0], mu[1], mu[2],
                inv_sigma[0], inv_sigma[1], inv_sigma[2]);
            
            // Check for error in kernel launch
            cudaError_t launch_err = cudaPeekAtLastError();
            if (launch_err != cudaSuccess) {
                LOG("kernel launch FAILED: %s", cudaGetErrorString(launch_err));
                return 1;
            }

            CUDA_CHECK(cudaDeviceSynchronize());
            // LOG("kernel sync done");
            
            if (i) {
                times.push_back(timer.toc_ms());
            }
        }
        
        // Calculate median
        std::sort(times.begin(), times.end());
        int median = times[times.size() / 2];
        medianTimes.push_back(median);
        
    }

    

    fpVal.copy_to_host(out);
    LOG("D2H done — %zu bytes copied", (size_t) N * sizeof(float));
    LOG("gpu out      [0..5] = %.4f %.4f %.4f %.4f %.4f %.4f",  out[0], out[1], out[2], out[3], out[4], out[5]);

    for (int i = 0; i < N; i++) {
        int c       = i % 3;
        expected[i] = (h_data[i] - mu[c]) * inv_sigma[c];
    }
    LOG("cpu expected [0..5] = %.4f %.4f %.4f %.4f %.4f %.4f", expected[0], expected[1], expected[2], expected[3], expected[4], expected[5]);

    int  first_mismatch = -1;
    int  mismatch_count = 0;
    for (int i = 0; i < N; i++) {
        if (out[i] != expected[i]) {
            if (first_mismatch < 0) first_mismatch = i;
            mismatch_count++;
        }
    }

    if (mismatch_count == 0) {
        LOG("PASS — all %d elements bit-identical to CPU reference", N);
    } else {
        LOG("FAIL — %d / %d mismatches; first at idx=%d channel=%d gpu=%.6f cpu=%.6f diff=%.6e",
            mismatch_count, N,
            first_mismatch, first_mismatch % 3,
            out[first_mismatch], expected[first_mismatch],
            out[first_mismatch] - expected[first_mismatch]);
    }

    for (size_t i = 0; i < blockSizes.size(); i++) {
        LOG("block size: %d, median time: %d ms", blockSizes[i], medianTimes[i]);
    }

    delete[] h_data;
    delete[] out;
    delete[] expected;
    return mismatch_count == 0 ? 0 : 1;
}

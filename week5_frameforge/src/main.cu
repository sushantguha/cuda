#include "frame.hpp"
#include "kernels.cuh"
#include "bench.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define WIDTH    1280
#define HEIGHT   960
#define CHANNELS 3
#define PIXELS   (WIDTH * HEIGHT * CHANNELS)

#define LOG(...) do { std::fprintf(stderr, "[host] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while(0)

constexpr float mu[3]        = { 0.485f, 0.456f, 0.406f };
constexpr float inv_sigma[3] = { 1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f };

int main(int argc, char** argv) {
    const int N    = PIXELS;
    int       runs = 50;
    for (int i = 1; i < argc - 1; i++) {
        if (std::strcmp(argv[i], "--runs") == 0) runs = std::atoi(argv[i + 1]);
    }

    // pinned host buffers (coalesced is guaranteed by the kernel itself)
    uint8_t* h_in;
    float*   h_out;
    CUDA_CHECK(cudaHostAlloc((void**)&h_in,  N * sizeof(uint8_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_out, N * sizeof(float),   cudaHostAllocDefault));
    for (int i = 0; i < N; i++) h_in[i] = i % 256;

    // v5 deploy = v4 at the empirically-optimal K=4, sum reduction unused
    const int K = 4;
    double    sum_unused = 0.0;

    // warm-up
    pipeline_v4_streams(h_in, h_out, &sum_unused, N, K,
        mu[0], mu[1], mu[2], inv_sigma[0], inv_sigma[1], inv_sigma[2]);

    std::vector<float> wall_times;
    wall_times.reserve(runs);
    for (int i = 0; i < runs; i++) {
        Timer t; t.tic();
        pipeline_v4_streams(h_in, h_out, &sum_unused, N, K,
            mu[0], mu[1], mu[2], inv_sigma[0], inv_sigma[1], inv_sigma[2]);
        wall_times.push_back(t.toc_ms());
    }
    std::sort(wall_times.begin(), wall_times.end());
    float wall_ms = wall_times[wall_times.size() / 2];

    LOG("v5 deploy (v4, K=%d)  runs=%d  wall(median): %.3f ms", K, runs, wall_ms);

    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    return 0;
}

#include "frame.hpp"
#include "kernels.cuh"
#include <cstdio>
#include "bench.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

#define WIDTH      1280
#define HEIGHT     960
#define CHANNELS   3
#define PIXELS     (WIDTH * HEIGHT * CHANNELS)
#define BLOCK_SIZE 256
#define RUNS       50

#define LOG(...) do { std::fprintf(stderr, "[host] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while(0)

constexpr float mu[3]        = { 0.485f, 0.456f, 0.406f };
constexpr float inv_sigma[3] = { 1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f };

int main() {
    const int   N            = PIXELS;
    const int   grid         = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const float kernel_bytes = (float)(N * 1 + N * 4);   // uint8 in + float32 out

    FrameBuffer<uint8_t> d_in(N);
    FrameBuffer<float>   d_out(N);

    // run_config: times kernel-only (→ effective BW) and H2D+kernel+D2H (→ e2e latency)
    auto run_config = [&](const char* label, auto kernel_fn, bool pinned) {
        uint8_t* h_in;
        float*   h_out;
        if (pinned) {
            void *raw_in, *raw_out;
            CUDA_CHECK(cudaHostAlloc(&raw_in,  N * sizeof(uint8_t), cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&raw_out, N * sizeof(float),   cudaHostAllocDefault));
            h_in  = static_cast<uint8_t*>(raw_in);
            h_out = static_cast<float*>(raw_out);
        } else {
            h_in  = new uint8_t[N];
            h_out = new float[N];
        }
        for (int i = 0; i < N; i++) h_in[i] = i % 256;

        // --- kernel-only bench (data pre-staged on device) ---
        d_in.copy_from_host(h_in);
        kernel_fn<<<grid, BLOCK_SIZE>>>(d_in.dataPtr, d_out.dataPtr, N,     // warm-up
            mu[0], mu[1], mu[2], inv_sigma[0], inv_sigma[1], inv_sigma[2]);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> k_times;
        k_times.reserve(RUNS);
        for (int i = 0; i < RUNS; i++) {
            Timer t; t.tic();
            kernel_fn<<<grid, BLOCK_SIZE>>>(d_in.dataPtr, d_out.dataPtr, N,
                mu[0], mu[1], mu[2], inv_sigma[0], inv_sigma[1], inv_sigma[2]);
            k_times.push_back(t.toc_ms());
        }
        std::sort(k_times.begin(), k_times.end());
        float k_ms = k_times[k_times.size() / 2];

        // --- e2e bench: H2D + kernel + D2H ---
        // warm-up
        CUDA_CHECK(cudaMemcpyAsync(d_in.dataPtr, h_in,  N * sizeof(uint8_t), cudaMemcpyHostToDevice));
        kernel_fn<<<grid, BLOCK_SIZE>>>(d_in.dataPtr, d_out.dataPtr, N,
            mu[0], mu[1], mu[2], inv_sigma[0], inv_sigma[1], inv_sigma[2]);
        CUDA_CHECK(cudaMemcpyAsync(h_out, d_out.dataPtr, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> e2e_times;
        e2e_times.reserve(RUNS);
        for (int i = 0; i < RUNS; i++) {
            Timer t; t.tic();
            CUDA_CHECK(cudaMemcpyAsync(d_in.dataPtr, h_in,  N * sizeof(uint8_t), cudaMemcpyHostToDevice));
            kernel_fn<<<grid, BLOCK_SIZE>>>(d_in.dataPtr, d_out.dataPtr, N,
                mu[0], mu[1], mu[2], inv_sigma[0], inv_sigma[1], inv_sigma[2]);
            CUDA_CHECK(cudaMemcpyAsync(h_out, d_out.dataPtr, N * sizeof(float), cudaMemcpyDeviceToHost));
            e2e_times.push_back(t.toc_ms());
        }
        std::sort(e2e_times.begin(), e2e_times.end());
        float e2e_ms = e2e_times[e2e_times.size() / 2];

        LOG("%-26s  kernel: %.3f ms  %.1f GB/s  |  e2e: %.3f ms",
            label, k_ms, kernel_bytes / k_ms / 1e6f, e2e_ms);

        if (pinned) { cudaFreeHost(h_in); cudaFreeHost(h_out); }
        else        { delete[] h_in; delete[] h_out; }
    };

    run_config("pinned   + coalesced", normalize_v2_launch,  true);
    run_config("pageable + coalesced", normalize_v2_launch,  false);
    run_config("pinned   + strided",   normalize_v1_strided, true);
    run_config("pageable + strided",   normalize_v1_strided, false);

    // ---------------- Day 4: streams + atomic reduction ----------------
    void *raw_in, *raw_out;
    CUDA_CHECK(cudaHostAlloc(&raw_in,  N * sizeof(uint8_t), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&raw_out, N * sizeof(float),   cudaHostAllocDefault));
    uint8_t* h_in_p  = static_cast<uint8_t*>(raw_in);
    float*   h_out_p = static_cast<float*>(raw_out);
    for (int i = 0; i < N; i++) h_in_p[i] = i % 256;

    // CPU reference: per-element normalized output + mean
    std::vector<float> cpu_out(N);
    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) {
        int c = i % 3;
        cpu_out[i] = ((float)h_in_p[i] - mu[c]) * inv_sigma[c];
        cpu_sum   += cpu_out[i];
    }
    double cpu_mean = cpu_sum / N;

    for (int K : {1, 2, 4, 8, 16, 32}) {
        std::vector<float> wall_times;
        wall_times.reserve(RUNS);
        double gpu_mean = 0.0;

        // warm-up + correctness check on per-element output
        pipeline_v4_streams(h_in_p, h_out_p, &gpu_mean, N, K,
            mu[0], mu[1], mu[2], inv_sigma[0], inv_sigma[1], inv_sigma[2]);

        float max_abs = 0.f, max_rel = 0.f;
        int   bad_idx = -1;
        for (int i = 0; i < N; i++) {
            float a   = std::fabs(h_out_p[i] - cpu_out[i]);
            float rel = a / std::max(std::fabs(cpu_out[i]), 1e-6f);
            if (a > max_abs)   { max_abs = a; bad_idx = i; }
            if (rel > max_rel) { max_rel = rel; }
        }

        for (int i = 0; i < RUNS; i++) {
            Timer t; t.tic();
            pipeline_v4_streams(h_in_p, h_out_p, &gpu_mean, N, K,
                mu[0], mu[1], mu[2], inv_sigma[0], inv_sigma[1], inv_sigma[2]);
            wall_times.push_back(t.toc_ms());
        }
        std::sort(wall_times.begin(), wall_times.end());
        float  wall_ms   = wall_times[wall_times.size() / 2];
        double mean_diff = std::fabs(gpu_mean - cpu_mean);

        LOG("K=%d  wall: %.3f ms  mean_|diff|=%.3e  out_max_abs=%.3e  out_max_rel=%.3e  (at i=%d)",
            K, wall_ms, mean_diff, max_abs, max_rel, bad_idx);
    }

    cudaFreeHost(h_in_p);
    cudaFreeHost(h_out_p);

    return 0;
}

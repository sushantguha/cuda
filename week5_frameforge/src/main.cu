#include "frame.hpp"
#include "kernels.cuh"
#include <cstdio>
#include "bench.hpp"
#include <vector>
#include <algorithm>

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

    // 8 MB > T4 L2 (4 MB) — used to flush cache between benchmarks
    void* l2_flush;
    CUDA_CHECK(cudaMalloc(&l2_flush, 8 * 1024 * 1024));
    auto flush_l2 = [&]() {
        CUDA_CHECK(cudaMemset(l2_flush, 0, 8 * 1024 * 1024));
        CUDA_CHECK(cudaDeviceSynchronize());
    };

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
        flush_l2();
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
        flush_l2();
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

    CUDA_CHECK(cudaFree(l2_flush));
    return 0;
}

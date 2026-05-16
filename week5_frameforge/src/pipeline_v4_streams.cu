#include "frame.hpp"
#include "kernels.cuh"
#include <algorithm>

void pipeline_v4_streams(const uint8_t* h_in, float* h_out, float* h_sum,
                         int N, int K,
                         float mu_r,        float mu_g,        float mu_b,
                         float inv_sigma_r, float inv_sigma_g, float inv_sigma_b) {
    if (K <= 0) return;
    if (K > N) return;
    const int BLOCK_SIZE = 256;

    FrameBuffer<uint8_t> d_in(N);
    FrameBuffer<float>   d_out(N);
    FrameBuffer<float>   d_sum(1);
    CUDA_CHECK(cudaMemset(d_sum.dataPtr, 0, sizeof(float)));

    cudaStream_t* streams = new cudaStream_t[K];
    for (int i = 0; i < K; i++) CUDA_CHECK(cudaStreamCreate(&streams[i]));

    int sizeOfEach = N / K;
    for (int i = 0; i < K; i++) {
        int offset     = i * sizeOfEach;
        int amountData = (i == K - 1) ? (N - offset) : sizeOfEach;

        d_in.copy_from_host_async(h_in + offset, amountData, offset, streams[i]);

        int realBlockSize = std::min(BLOCK_SIZE, amountData);
        int grid          = (amountData + realBlockSize - 1) / realBlockSize;

        normalize_v3_coalesced<<<grid, realBlockSize, 0, streams[i]>>>(
            d_in.dataPtr + offset, d_out.dataPtr + offset, amountData,
            mu_r, mu_g, mu_b, inv_sigma_r, inv_sigma_g, inv_sigma_b,
            d_sum.dataPtr);

        d_out.copy_to_host_async(h_out + offset, amountData, offset, streams[i]);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < K; i++) CUDA_CHECK(cudaStreamDestroy(streams[i]));
    delete[] streams;

    CUDA_CHECK(cudaMemcpy(h_sum, d_sum.dataPtr, sizeof(float), cudaMemcpyDeviceToHost));
    *h_sum /= N;
    // d_in / d_out / d_sum freed by FrameBuffer destructors
}

#include "frame.hpp"
#include "kernels.cuh"
#define WIDTH 1280
#define HEIGHT 960
#define CHANNELS 3
#define PIXELS WIDTH * HEIGHT * CHANNELS
#define BLOCK_SIZE 256

constexpr float mu[3]    = { 0.485f, 0.456f, 0.406f };
constexpr float inv_sigma[3] = { 1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f };

int main() {
    FrameBuffer<uint8_t> uint8Val(PIXELS);
    FrameBuffer<float> fpVal(PIXELS);

    uint8_t* h_data = new uint8_t[PIXELS];
    float* out = new float[PIXELS];

    // Initialize the uint8 buffer with some test data
    for (int i = 0; i < PIXELS; i++) {
        h_data[i] = i % 256;
    }

    // Copy input data to device.
    uint8Val.copy_from_host(h_data);
    
    // Call kernel to process the data.
    normalize_v1_naive<<<(PIXELS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(uint8Val.dataPtr, fpVal.dataPtr, PIXELS, mu[0], mu[1], mu[2], inv_sigma[0], inv_sigma[1], inv_sigma[2]);
    
    cudaDeviceSynchronize();

    // Copy result to host.
    fpVal.copy_to_host(out);

    // Compute the expected result on host
    float* expected = new float[PIXELS];
    for (int i = 0; i < PIXELS; i++) {
        int channel = i % 3;
        expected[i] = (h_data[i] - mu[channel]) * inv_sigma[channel];
    }

    // Compare result with host-computation.
    bool passed = true;
    for (int i = 0; i < PIXELS; i++) {
        if (out[i] != expected[i]) {
            passed = false;
            break;
        }
    }
    
    if (passed) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    
    return 0;
}
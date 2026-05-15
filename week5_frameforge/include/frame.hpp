#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t err__ = (expr);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                    \
                   cudaGetErrorName(err__), __FILE__, __LINE__,               \
                   cudaGetErrorString(err__));                                \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

template<typename T>
class FrameBuffer {
    public:
    T* dataPtr;
    size_t size;
    FrameBuffer(size_t n): dataPtr(nullptr), size(n) {
        CUDA_CHECK(cudaMalloc((void**) &dataPtr, n * sizeof(T)));
    }
    ~FrameBuffer() {
       if (dataPtr) CUDA_CHECK(cudaFree(dataPtr));
    }

    FrameBuffer(const FrameBuffer&) = delete;
    FrameBuffer& operator=(const FrameBuffer&) = delete;

    FrameBuffer(FrameBuffer&& other) noexcept : dataPtr(other.dataPtr) {
        other.dataPtr = nullptr;
        size = other.size;
        other.size = 0;
    }
    
    FrameBuffer& operator=(FrameBuffer&& other) noexcept {
        if (this != &other) {
            if (dataPtr) CUDA_CHECK(cudaFree(dataPtr));
            dataPtr = other.dataPtr;
            other.dataPtr = nullptr;
            size = other.size;
            other.size = 0;
        }
        return *this;
    }
    void copy_from_host(const T* h_src) {
        CUDA_CHECK(cudaMemcpy(dataPtr, h_src,
                            size * sizeof(T),
                            cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* h_dst) const {
        CUDA_CHECK(cudaMemcpy(h_dst, dataPtr,
                            size * sizeof(T),
                            cudaMemcpyDeviceToHost));
    }
    T* device() noexcept {
        return dataPtr;
    }
    size_t getSize() const noexcept {
        return size;
    }
};
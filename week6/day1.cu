#include <iostream>

__global__ void branch() {
    if (threadIdx.x % 2 == 0) {
        printf("Even thread: %d\n", threadIdx.x);
    } else {
        printf("Odd thread: %d\n", threadIdx.x);
    }
    __syncwarp();
    printf("Completed thread: %d\n", threadIdx.x);
}

int main() {
    branch<<<1, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}

#include <iostream>

__global__ void branch() {
    if (threadIdx.x % 2 == 0) {
        // printf("Even thread: %d\n", threadIdx.x);
        int a = 1;
    } else {
        // printf("Odd thread: %d\n", threadIdx.x);
        int b = 2;
    }
    __syncwarp();
    // printf("Completed thread: %d\n", threadIdx.x);
    int c = 3;
}

int main() {
    branch<<<1, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}

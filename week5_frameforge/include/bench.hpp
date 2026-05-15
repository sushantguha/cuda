struct Timer {
  cudaEvent_t start, stop;
  Timer()  { cudaEventCreate(&start); cudaEventCreate(&stop); }
  ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
  void tic()             { cudaEventRecord(start); }
  float toc_ms() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
  }
};
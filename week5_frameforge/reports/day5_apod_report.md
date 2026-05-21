# Day 5 — APOD Report: FrameForge

**Workload:** simulated FSD frame preprocessor — `uint8` 1280×960×3 → `float32` 1280×960×3 normalize (`y = (x − μ_c) · inv_σ_c`).
**Device:** NVIDIA T4 (CC 7.5), memory clock 5001 MHz, 256-bit bus → **320.1 GB/s** theoretical peak.

---

## A — Assess

Static analysis: each output element reads 1 byte (uint8 in) and writes 4 bytes (float32 out) = **5 B/element**, and computes one subtract + one multiply = **2 FLOPs/element**. Arithmetic intensity ≈ **0.4 FLOP/B**, well below the T4's ~13 FLOP/B ridge point. This is a **memory-bound kernel** — adding compute parallelism cannot help; the bandwidth roofline is the ceiling.

Confirmed by ncu (Day 5): `dram__throughput.avg.pct_of_peak_sustained_elapsed` = **42.6 %** of the 320 GB/s peak. SMs spend ~19.5k cycles per chunk waiting on DRAM, not crunching numbers.

## P — Parallelize

Day 2 block-size sweep, one thread per output element, kernel held constant:

| block | 32 | 64 | 128 | 256 | 512 | 1024 |
|------:|---:|---:|----:|----:|----:|-----:|
| ms    |0.270|0.125|0.117|0.117|0.118|0.124|

Empirical optimum: **256** (tied with 128, picked for warp-aligned cleanliness). At block=32 only 16 warps/SM can be resident — insufficient latency-hiding parallelism. At block=1024 the SM can hold only one block, which restricts the scheduler's flexibility. The 128–512 plateau is the full-occupancy regime on T4 (32 warps/SM).

## O — Optimize

Three orthogonal optimizations applied on top of the v2 launch config:

| Variant                    | Kernel ms | Effective BW | % of 320 GB/s peak | e2e ms |
|---------------------------|----------:|-------------:|-------------------:|-------:|
| v2 strided + pageable      |   0.358   |   51.5 GB/s  |        16 %        | 4.619  |
| v2 strided + pinned        |   0.437   |   42.2 GB/s  |        13 %        | 1.875  |
| v3 coalesced + pageable    |   0.113   |  163.4 GB/s  |        51 %        | 4.164  |
| **v3 coalesced + pinned**  | **0.129** |**142.9 GB/s**|      **45 %**      |**1.553**|

Two findings:

1. **Coalescing dominates kernel throughput** — laying out memory so consecutive threads in a warp read consecutive bytes turns 4 sector loads per warp into 1, giving a **~3.4× kernel speedup** (strided 42 → coalesced 143 GB/s). Pinned vs pageable does *not* affect kernel time, because once the data is on the device the host transfer is finished.
2. **Pinned host memory dominates e2e** — pinned cuts e2e from ~4.2 ms to ~1.55 ms (**~2.7×**), because pageable transfers force the driver to stage through an internal pinned buffer (two host copies instead of one) before DMA can run.

Day 4 added **multi-stream pipelining** with `atomicAdd` reduction. K-sweep wall times (median of 50 runs, including per-call alloc/stream-create overhead inside the timed region):

| K | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|--:|--:|--:|--:|--:|---:|---:|---:|----:|----:|
| ms |2.364|2.208|**2.149**|2.162|2.204|2.288|2.538|3.037|5.101|

Optimum at **K=4**. Beyond K≈16, per-chunk launch overhead (~20 µs × K from `cudaStreamCreate`/`cudaMemcpyAsync` queueing) starts to outweigh the overlap benefit. The theoretical ceiling on this workload is set by D2H (the float32 output is 4× the uint8 input), giving a hard floor of ~1.13 ms — the measured K=4 wall is above that because alloc/stream-create are inside the timed region.

## D — Deploy

Final config baked into the binary:

- Block size = 256, grid = `ceil(N / 256)`
- Coalesced load pattern (`tid = blockIdx.x*blockDim.x + threadIdx.x`, modulo-3 channel select)
- `cudaHostAlloc` pinned host buffers
- K = 4 streams, async H2D + kernel + D2H per chunk
- `atomicAdd` reduction dropped (the downstream model computes its own statistics; the Day 4 atomicAdd was a teaching exercise, not a deploy requirement)

**ncu profile** (`reports/day5_ncu.csv`, 8 launches captured):

| Metric | Median per launch | Notes |
|---|---:|---|
| `dram__bytes_read.sum`  | 1.09 MB | vs. 0.92 MB theoretical → ~19 % overhead from L2 sector fills |
| `dram__bytes_write.sum` | 3.14 MB | vs. 3.69 MB theoretical → **L2 absorbed ~15 % of writes** before DRAM |
| `dram__throughput…pct_of_peak` | **42.6 %** | the headline number — physical DRAM utilization |
| `sm__cycles_elapsed.avg` | 19,540 cycles | ~13 µs per chunk at 1.5 GHz boost |

Variance across launches < 2 %, so the measurement is stable.

### Reproducibility

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/frameforge --runs 50                          # wall-time benchmark
ncu --target-processes all \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --csv ./build/frameforge --runs 1 > reports/day5_ncu.csv
```

### Where the headroom is (Week 6 preview)

42.6 % of peak DRAM is honest territory for a naive coalesced kernel. The remaining 57 % is reachable via Week 6 techniques explicitly out of scope for Week 5:

- Vectorized `uchar4` / `float4` loads — fewer, wider transactions, typically pushes memory-bound kernels from ~40 % to ~70 % of peak.
- Warp-shuffle reductions (`__shfl_down_sync`) — replaces the serialized `atomicAdd` path if the reduction is re-added.
- `__constant__` memory for μ/σ — frees a few registers, marginal on this workload.

### Tesla tie-in

At 36 FPS × 8 cameras the FSD chip has ~3.5 ms per frame of preprocessing budget per camera. v5 hits **~2.1 ms wall** (with alloc inside the timed region) and the kernel alone is **0.13 ms** — the full preprocessor fits comfortably inside the per-camera budget on a single T4-class accelerator, with the dominant remaining cost being PCIe D2H transfer, not compute.

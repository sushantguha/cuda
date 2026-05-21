# Week 5 Capstone Project — **FrameForge**
### A from-scratch CUDA preprocessor for a simulated FSD camera stream

> **Curriculum context:** Tesla AI Hardware Study Guide 2026 — Phase 2 (CUDA), Week 5 "GPU Architecture & Basics".
> **Goal:** Implement *one* end-to-end CUDA project whose scope matches **exactly what the Week 5 resources actually teach** — kernel launch syntax, the thread/block/grid hierarchy, the APOD framework, the memory hierarchy + coalesced access, pinned memory + streams, atomics, and event-based timing for theoretical vs effective bandwidth.
> **Duration:** 5 days (≈ 2–3 hrs/day). Single GitHub repo. Each commit ties one Week 5 concept to one measurable number.
> **Scope honesty:** Warp-shuffle reductions, `__ballot_sync`, bank-conflict-free shared memory, `__constant__` memory, vectorized `float4` loads, and Nsight Compute roofline plots are **explicitly out of scope for Week 5** and listed as Week-6-preview stretch goals at the bottom. They are not taught in the listed Week 5 resources.

---

## 1. Resources Actually Read This Week (and What Each One Covers)

| Day | Resource | What it actually teaches |
| --- | --- | --- |
| D1 | **NVIDIA CUDA C++ Programming Guide, Ch 1–3** | Prose-only overview: why GPUs, scalable programming model. No code. Background reading. |
| D1 | **freeCodeCamp CUDA Course, first ~3 hrs** (Modules 1–5: Deep-Learning ecosystem → setup → C/C++ refresher → "Gentle Intro to GPUs" → "Writing your First Kernels" — vector add v1, v2, naive matmul, intro profiling, atomics, streams) | Kernel launch syntax `<<<grid, block>>>`, `__global__`/`__device__`, `cudaMalloc`/`cudaMemcpy` vs `cudaMallocManaged`, `cudaDeviceSynchronize`, `blockIdx/threadIdx/blockDim/gridDim`. SM ↔ block ↔ thread mapping at conceptual level. |
| D2 | **NVIDIA CUDA C++ Best Practices Guide, Ch 1–3** | The **APOD** framework (Assess → Parallelize → Optimize → Deploy). CPU vs GPU architectural trade-offs. Recommendations are prioritized by impact. |
| D2 | **freeCodeCamp hours 4–5** | Same Module 5 — deeper passes over kernel structure, intro profiling with `nvprof`/`ncu` at the *command-line* level (not Nsight UI), atomics, streams. |
| D3 | **Programming Guide Ch 5 (Memory Hierarchy)** | §5.1 Kernels, §5.2 Thread Hierarchy, §5.3 Memory Hierarchy, §5.4 Heterogeneous Programming, §5.5 Async SIMT, §5.6 Compute Capability. Names every memory space (registers, local, shared, global, constant, texture) and the ownership / scope of each. |
| D3 | **Best Practices Ch 9 (Performance Metrics)** | Event-based timing (`cudaEvent_t`), CPU-timer synchronization requirements, **theoretical bandwidth** (memory-clock × bus-width × 2 / 8), **effective bandwidth** (`(bytes_read + bytes_written) / elapsed`), profiler-metric interpretation. *Does not cover launch-config tuning or occupancy.* |
| D4 | **Best Practices Ch 10 (Memory Optimizations)** | Host↔device transfers, **pinned (page-locked) host memory**, **asynchronous copies + streams**, the device memory spaces in detail, **coalesced global memory access** (alignment, strides), shared-memory **bank conflicts at the conceptual level**, L2-cache persistence, register pressure. *Does not cover warp-level primitives or occupancy.* |
| D4 | **Programming Guide §6 "Programming Interface"** (the subsection on warp-level intrinsics is referenced but content lives at §10.19/§10.22, outside the listed reading) | Warp-level primitives (`__shfl_sync`, `__ballot_sync`) are *named* in Week 5 but their tutorial-level coverage is **Week 6 territory**. We acknowledge them and stop there. |
| D5 | Weekly review + `ncu` / `nvprof` command-line | Capture effective bandwidth and basic kernel metrics; write a one-page APOD-structured report. |

The project below is scoped *tightly* to what is listed above. Anything that requires resources not on this list is moved to §7 Stretch Goals.

---

## 2. The Problem (Tesla Tie-In)

A Tesla FSD camera streams **8 cameras × 1280 × 960 × 3 channels @ 36 FPS** into the FSD chip. Before any neural network runs, every frame must be:

1. **Normalized** — convert `uint8` pixels to `float32` in `[-1, 1]` using a per-channel affine `y = (x − μ_c) / σ_c`.
2. **Aggregated** — produce a per-frame **mean luminance** scalar used by the downstream auto-exposure controller.

This is the simplest possible "real" GPU preprocessing pipeline: a pointwise transform followed by a sum reduction. It is also the canonical Week 5 teaching workload because it fits *cleanly inside what the resources actually teach*: launch a kernel, copy data in, copy data out, measure bandwidth, time it, atomically accumulate a scalar.

```
   uint8 frame  ─┐
   (1280×960×3) │   ┌──────────────────┐    float32 frame ─┐
                ├──▶│  Normalize (FMA) │──▶ (1280×960×3)   │   ┌──────────────────────┐    mean_luminance
   μ_c, σ_c   ──┘   └──────────────────┘                   ├──▶│  Reduce (atomicAdd)  │──▶  (1 float / frame)
   (kernel args)                                          (host or device)
```

---

## 3. What You Will Build

A single CMake project: `frameforge/`

```
frameforge/
├── CMakeLists.txt
├── README.md                      # auto-grown each day with one benchmark row + an APOD note
├── include/
│   ├── frame.hpp                  # FrameBuffer<T> RAII wrapper (host + device)
│   ├── kernels.cuh                # extern declarations for every kernel variant
│   └── bench.hpp                  # cudaEvent_t-based timing harness
├── src/
│   ├── main.cu                    # CLI: --kernel <v1..v5> --block <N> --runs <K>
│   ├── frame.cu                   # FrameBuffer impl (cudaMalloc + pinned host via cudaHostAlloc)
│   ├── reference_cpu.cpp          # serial ground-truth (correctness oracle)
│   ├── normalize_v1_naive.cu      # Day 1
│   ├── normalize_v2_launch.cu     # Day 2 — same kernel, launch-config sweep
│   ├── normalize_v3_coalesced.cu  # Day 3 — coalesced loads + pinned memory + effective BW
│   ├── pipeline_v4_streams.cu     # Day 4 — multi-stream overlap + atomic reduction
│   └── pipeline_v5_apod.cu        # Day 5 — final form, profiled, APOD report attached
├── scripts/
│   ├── sweep_launch.sh            # Day 2 — block-size sweep
│   └── ncu_collect.sh             # Day 5 — Nsight Compute / nvprof metric capture (CLI only)
└── reports/
    ├── day1_launch_indices.md     # printf-dumped launch geometry, annotated
    ├── day2_apod_assess.md        # APOD "Assess" step, baseline numbers
    ├── day3_bandwidth.md          # theoretical vs effective bandwidth table
    ├── day4_streams.md            # timeline notes, atomic-reduction correctness
    └── day5_apod_report.md        # the one-page final write-up
```

Hard rules:
- **No third-party CUDA libraries** (no Thrust, CUB, cuBLAS). Hand-roll everything Week 5 requires.
- Every kernel must be **numerically verified** against `reference_cpu.cpp` to within `1e-4` relative error before timing.
- Every benchmark reports **(median, P99) of ≥ 50 runs** after one warm-up iteration.
- Every commit message follows `[wk5 dN] <topic>: <one-line result>` — e.g. `[wk5 d3] coalesced loads: 78 → 240 GB/s effective`.

---

## 4. Day-by-Day Plan

Each day's tasks are scoped to what the listed resources actually teach. A "Resource check" line at the start of each day cites exactly where in the reading the technique comes from.

### Day 1 — Kernel Launch Syntax, Thread Hierarchy
**Maps to:** W5 D1 — *GPU architecture; SM, threads/blocks/grids/warps; kernel launch syntax.*
**Resource check:** Programming Guide §5.1 (Kernels), §5.2 (Thread Hierarchy); freeCodeCamp Module 5 / "01 CUDA Basics" + "02 Kernels / 00_vector_add_v1.cu".

**Tasks**
1. Set up CMake with `enable_language(CUDA)`, `CUDA_ARCHITECTURES native`, two configs: `Debug` (`-G`) and `Release` (`-O3 -lineinfo`).
2. Implement `FrameBuffer<T>` — RAII wrapper over `cudaMalloc` / `cudaFree`. Allocate one device-side `uint8` frame and one device-side `float` frame.
3. Implement `normalize_v1_naive`. Required signature:
   ```cuda
   __global__ void normalize_v1_naive(const uint8_t* __restrict__ in,
                                      float*          __restrict__ out,
                                      int   N,
                                      float mu_r,  float mu_g,  float mu_b,
                                      float inv_sigma_r, float inv_sigma_g, float inv_sigma_b);
   ```
4. **Inside the kernel**, on the *first thread of the first block only*, `printf` the launch geometry: `gridDim.x`, `blockDim.x`, `blockIdx.x`, `threadIdx.x`, `warpSize`, and the global thread id. (Per Module 5's "understand the concept of threads, blocks, and grids" practice.)
5. Verify numerically against `reference_cpu.cpp` on a 1280×960×3 frame.

**Deliverable**
- `reports/day1_launch_indices.md` — paste the printf output. Annotate which dimension is which. In your own words (3–5 sentences) describe the SM → block → thread mapping as taught in Module 5's "Hardware Mapping" section. Mention that warps are 32 threads wide and that you'll come back to them later — Week 5 stops at *naming* them.

**Acceptance**
- [ ] Kernel compiles for `native` arch, passes correctness check.
- [ ] Annotated index dump committed.
- [ ] One-sentence Tesla tie-in in the commit body.

---

### Day 2 — APOD: Assess + Parallelize. Block-Size Sweep.
**Maps to:** W5 D2 — *Execution model, thread scheduling, global memory access; vary block/grid sizes, measure throughput.*
**Resource check:** Best Practices Ch 1–3 (APOD framework); Programming Guide §5.2 (Thread Hierarchy) again; freeCodeCamp profiling intro (CLI `nvprof` / `ncu --print-summary`).

**Important honesty:** the Best Practices Guide does *not* prescribe a particular block size or formal occupancy calculation in Ch 1–3 or Ch 9. So Day 2 is **about applying the APOD method**, not about chasing occupancy.

**Tasks**
1. Copy `normalize_v1_naive` → `normalize_v2_launch.cu` **with no algorithmic change**. The only variable today is *how you launch it*.
2. Run an **A**ssess step (APOD "A"): identify, in writing, that this kernel is memory-bound (you read 3 bytes + write 12 bytes per output element ⇒ ~15 B/output; compute = a small constant per output, so arithmetic intensity is < 1 FLOP/byte). Commit a short paragraph saying so.
3. Run a **P**arallelize sweep (APOD "P"): test block sizes ∈ {32, 64, 128, 256, 512, 1024} with grid size = `ceil_div(N, block)` (one thread per element). For each, record median latency over 50 runs.
4. Write the results to `reports/day2_apod_assess.md` as a markdown table. Find the best block size empirically. **Do not** try to derive it from occupancy — that's Week 7 (Phase 2 D5 of the study guide says Nsight occupancy report, but the *Ch 9 reading explicitly does not cover occupancy*, so we stay descriptive).

**Acceptance**
- [ ] Markdown table with ≥ 6 block-size data points.
- [ ] One paragraph naming this an APOD "Assess + Parallelize" pass.
- [ ] README updated with "Day 2 best block size = N".

---

### Day 3 — Memory Hierarchy, Coalescing, Theoretical vs Effective Bandwidth
**Maps to:** W5 D3 — *Shared memory, constant memory, memory coalescing; measure bandwidth with `nvprof` or Nsight.*
**Resource check:** Programming Guide §5.3 (Memory Hierarchy — names every space); Best Practices Ch 9 (theoretical bandwidth formula + effective-bandwidth formula); Best Practices Ch 10 (coalesced global memory access, alignment & strides).

**Honesty note:** the study-guide line *"Optimize vector add with coalesced access"* is actually well covered by Best Practices Ch 10's coalescing rules. **But** `__constant__` memory and shared-memory tiling are barely covered in the Week 5 reading (named in §5.3, not tutorialized). We therefore implement **coalescing** for real and only *mention* shared/constant memory in the Day 3 report — they become stretch goals.

**Tasks**
1. Create `normalize_v3_coalesced.cu`. Required change vs v2: lay out the input and output so that **thread `t` of block `b` reads byte index `(b * blockDim.x + t)`** — i.e. consecutive threads in a warp read consecutive bytes. (You're already doing this if you indexed by `idx = blockIdx.x * blockDim.x + threadIdx.x` on Day 1 — Day 3 is about *verifying* it with measurement, not changing the algorithm. If your Day 1 indexing was stride-based, fix it here.)
2. Switch the host-side allocation from pageable `malloc` to **pinned memory** via `cudaHostAlloc(..., cudaHostAllocDefault)`. Best Practices Ch 10 §10.1.1.
3. Use the **theoretical bandwidth formula** from Best Practices §9.1 to compute your device's peak BW:
   `BW_theoretical = (mem_clock_kHz * 2 * mem_bus_bits / 8) / 1e6  // GB/s`
   Read `mem_clock_kHz` and `mem_bus_bits` from `cudaGetDeviceProperties`. Print both.
4. Compute **effective bandwidth** per Best Practices §9.2 using your event-timed runs:
   `BW_effective = (bytes_read + bytes_written) / elapsed_ms / 1e6  // GB/s`
   where `bytes_read = N` (uint8 in) and `bytes_written = 4 * N` (float32 out).
5. Compare v1 vs v2 vs v3, with and without pinned host memory (so 4 rows total: pageable+naive, pinned+naive, pageable+coalesced, pinned+coalesced).

**Deliverable**
- `reports/day3_bandwidth.md` — table with 4 rows × {latency ms, GB/s effective, % of theoretical}. One paragraph on what *coalescing* means at the memory-controller level (Ch 10 §10.2.1). One paragraph on why **pinned memory** matters for the *host transfer* even though the kernel itself doesn't care.
- A note (~3 sentences) saying: "Shared memory and constant memory are named in §5.3 but not tutorialized in this week's readings. They are Day-3 stretch goals."

**Acceptance**
- [ ] v3 numerically equal to v1.
- [ ] Theoretical BW computed from real device properties, not a hardcoded peak.
- [ ] Effective BW for v3 ≥ 50 % of theoretical (this is a generous target; tighter targets need Week 6 techniques).

---

### Day 4 — Streams + Atomics (the reduction)
**Maps to:** W5 D4 — *Shared memory bank conflicts, warp-level primitives.*
**Resource check:** **Mismatch.** The study guide nominally lists warp-shuffle primitives for Day 4, but the listed reading (Programming Guide on warps, Best Practices Ch 10) tutorializes **streams, async copies, pinned memory, and bank conflicts at a conceptual level** — *not* `__shfl_sync` / `__ballot_sync` patterns. freeCodeCamp Module 5 covers **atomics + streams** here, not warp primitives. We honor what the resources actually teach: streams + atomics today, warp primitives are Week 6 preview.

**Tasks**
1. Implement `pipeline_v4_streams.cu`. Architecture:
   - Split the 1280×960×3 frame into **K chunks** along the row dimension.
   - Create **K CUDA streams** (`cudaStreamCreate`).
   - For each chunk `k`: async H2D copy → launch `normalize_v3_coalesced` on stream `k` → async D2H copy back. Plus a reduction kernel that accumulates `mean_luminance` per chunk.
   - The reduction uses **`atomicAdd`** on a single device-side `float` accumulator. (Module 5 → Atomics, intro level. We avoid warp-shuffle reductions — those need `__shfl_down_sync`, which is Week 6.)
   - Final mean = device accumulator / N, computed on the host.
2. Compare against a single-stream baseline (chunks = 1). Measure end-to-end wall time including H2D + kernel + D2H.
3. **Verify** `atomicAdd` correctness: run with K = {1, 2, 4, 8} and confirm the mean luminance is bit-identical to CPU reference to within 1e-3 (floating point summation order matters — note this, do not panic).

**Deliverable**
- `reports/day4_streams.md` — table with K ∈ {1, 2, 4, 8} streams showing wall time, atomic correctness check, and a one-paragraph explanation of *how streams hide H2D/D2H latency behind compute* (Best Practices §10.1).
- One paragraph: "Why we are **not** using `__shfl_down_sync` for the reduction this week — that primitive is not tutorialized by any of the listed Week 5 resources. Naive `atomicAdd` is the resource-faithful choice; warp-shuffle reduction is a Week 6 stretch goal."

**Acceptance**
- [ ] Multi-stream version faster than single-stream by at least 1.2× on a frame ≥ 4 MB.
- [ ] `atomicAdd`-based mean numerically equal to CPU reference within 1e-3.
- [ ] `compute-sanitizer --tool racecheck` clean.

---

### Day 5 — APOD: Optimize + Deploy. CLI Profiling. One-Page Report.
**Maps to:** W5 D5 — *Weekly review + Nsight profiling.*
**Resource check:** Best Practices Ch 2 (APOD final two steps: O and D); Best Practices §9.2 (effective-bandwidth-as-truth metric); freeCodeCamp Module 5 / "03 Profiling".

**Honesty note:** the freeCodeCamp profiling segment is *command-line* `nvprof` / `ncu --print-summary` level — it does *not* teach the Nsight Compute GUI or roofline modeling. Day 5 sticks to CLI metrics. Roofline plots are Week 7 territory (study guide's own Roofline Day is Week 13 D3).

**Tasks**
1. Create `pipeline_v5_apod.cu` — your **final** form. It's just v4 (streams + atomic reduction) with any cleanup, plus a single `--kernel v5` switch in `main.cu` that runs the canonical configuration: best block size from Day 2, coalesced + pinned (Day 3), K = best-stream-count from Day 4.
2. Run `scripts/ncu_collect.sh`:
   ```bash
   ncu --target-processes all \
       --metrics dram__bytes_read.sum,dram__bytes_write.sum,sm__cycles_elapsed.avg,dram__throughput.avg.pct_of_peak_sustained_elapsed \
       --csv ./build/frameforge --runs 1 > reports/day5_ncu.csv
   ```
   (Keep the metric list on one line — splitting it across lines injects whitespace into the comma-list, which ncu then mistakes for an executable argument.)
   (`--runs 1` because ncu replays each kernel internally to gather metrics; an in-code timing loop of 50 just multiplies profile time without adding signal.)
   (If `ncu` is unavailable on the dev machine, fall back to `nvprof --metrics achieved_occupancy,gld_throughput,gst_throughput`.)
3. Write **`reports/day5_apod_report.md`** — one printed page, structured as the four APOD letters:
   - **A**ssess — what bottleneck did profiling reveal? (Memory-bound vs compute-bound — Best Practices §3.)
   - **P**arallelize — block-size sweep result + naive vector implementation.
   - **O**ptimize — coalesced access + pinned memory + streams + atomic reduction. Show the speedup curve v1 → v5.
   - **D**eploy — final benchmark table; reproducibility note ("`cmake --build build && ./build/frameforge --kernel v5 --runs 50` reproduces every number in this report").
   - **One-sentence Tesla tie-in** at the bottom.

**Acceptance**
- [ ] `pipeline_v5_apod` is the fastest end-to-end version measured.
- [ ] CSV or text profile output committed (whichever your tooling produces).
- [ ] One-page report covers all four APOD letters by name.

---

## 5. Architecture & Code Standards

- **C++ standard:** C++20 (`-std=c++20`).
- **CUDA standard:** `--std c++17` on the device side.
- **Sanitizers:** Build a third config `Sanitize` (`-fsanitize=address,undefined` on host code); host harness must be clean.
- **`compute-sanitizer` (the CUDA equivalent of valgrind):** every kernel must pass `compute-sanitizer --tool memcheck` and `--tool racecheck` clean.
- **Error handling:** wrap every CUDA API call in a `CUDA_CHECK(...)` macro that prints file/line on failure.
- **Determinism:** seed the host-side random frame generator (`std::mt19937{42}`) so benchmarks are reproducible commit-to-commit.

### `CUDA_CHECK` macro

```cpp
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
```

### Timing harness (used by every benchmark)

```cpp
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
```

This matches Best Practices §9.1.2 ("Using CUDA GPU Timers") exactly.

---

## 6. Final Deliverable Checklist (end of Day 5)

- [ ] Single repo `frameforge/` builds cleanly with `cmake -S . -B build && cmake --build build`.
- [ ] `./build/frameforge --kernel v1` through `v5` all run and report timing.
- [ ] Five reports (one per day) in `reports/`.
- [ ] `reports/day5_apod_report.md` final write-up with APOD-letter structure.
- [ ] All commits follow the `[wk5 dN] ...` convention.
- [ ] `compute-sanitizer` clean on all kernels.
- [ ] **README ends with a results table** (numbers will vary by GPU):

| Kernel | Concept introduced                | Latency (ms) | Effective BW (GB/s) | % of theoretical peak | Speedup vs v1 |
| ------ | --------------------------------- | -----------: | -------------------: | --------------------: | ------------: |
| v1     | naive, one-thread-per-pixel       |       _xx.x_ |                _xxx_ |                _xx %_ |          1.0× |
| v2     | best block size (APOD Assess+P)   |       _xx.x_ |                _xxx_ |                _xx %_ |           _×_ |
| v3     | coalesced + pinned host memory    |       _xx.x_ |                _xxx_ |                _xx %_ |           _×_ |
| v4     | multi-stream + `atomicAdd` reduce |       _xx.x_ |                _xxx_ |                _xx %_ |           _×_ |
| v5     | **final form (APOD-deployed)**    |       _xx.x_ |                _xxx_ |                _xx %_ |       **_×_** |

---

## 7. Stretch Goals — *Week 6+ Preview, Not Required for Week 5*

These were in the earlier draft of this project and are honestly out of scope for what the **Week 5 resources** actually teach. They are good warm-ups for Week 6 (the study guide's "Divergence, Synchronization & Reduction Patterns" week, which *does* tutorialize warp shuffles in its own resources).

1. **Warp-shuffle reduction** — replace the `atomicAdd` in v4 with a `__shfl_down_sync`-based intra-warp reduction, then a `atomicAdd` once per block. *Needs Programming Guide §10.22 and the NVIDIA blog "Faster Parallel Reductions on Kepler"* — both are Week 6 D3 resources, not Week 5.
2. **`__ballot_sync` for saturated highlights** — count threads whose post-normalize value exceeds `+1.0`. Same Week 6 caveat.
3. **Bank-conflict-free shared-memory reduction** — pad-by-1 trick. Best Practices §10.2.3 names bank conflicts but does not walk through the padding fix as an exercise. Week 6.
4. **`__constant__` memory for μ/σ** — bind the six normalize constants via `cudaMemcpyToSymbol`. Programming Guide §5.3.5 names constant memory; tutorial-level coverage is sparse in Week 5 resources.
5. **Vectorized `uchar4` / `float4` loads** — covered by freeCodeCamp Module **7** ("Faster Matmul"), not Module 5.
6. **Nsight Compute roofline plots** — covered at NVIDIA developer-blog depth; the listed Best Practices Ch 9 doesn't include roofline modeling. Study guide's own roofline day is Week 13 D3.
7. **Occupancy calculator via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`** — *not* in Best Practices Ch 9. Module 7 of freeCodeCamp introduces occupancy in the context of matmul optimization.

If you finish Days 1–5 in fewer than 5 days and want to keep going, pick one of these and add it as a `v6` variant — but **mark it explicitly as "Week 6 preview" in the commit message**, not as Week 5 work.

---

## 8. Week 5 Topic ↔ Project Mapping (proof every *in-scope* topic is covered)

| Week 5 Topic (what the resources actually teach)         | Where it appears                                  |
| -------------------------------------------------------- | ------------------------------------------------- |
| Kernel launch syntax `<<<grid, block>>>`                 | Day 1 — `normalize_v1_naive`                      |
| Threads, blocks, grids, SM↔block mapping                 | Day 1 — printf launch geometry                    |
| Warps named (32 threads) — *concept only*                | Day 1 — printf includes `warpSize`                |
| APOD framework (Assess / Parallelize / Optimize / Deploy) | Day 2 (A+P) + Day 5 (O+D)                         |
| Block-size sweep / empirical launch tuning               | Day 2 — `sweep_launch.sh`                         |
| Memory hierarchy (registers, local, shared, global, constant, texture) | Day 3 — named in report; Day 3+ uses global + pinned host |
| Coalesced global memory access                           | Day 3 — `normalize_v3_coalesced`                  |
| Pinned (page-locked) host memory                         | Day 3 — `cudaHostAlloc`                           |
| Theoretical vs effective bandwidth                       | Day 3 — formulas from Best Practices §9.1–§9.2    |
| CUDA streams + async copies                              | Day 4 — `pipeline_v4_streams`                     |
| Atomics (`atomicAdd`)                                    | Day 4 — frame-mean reduction                      |
| Event-based timing                                       | All days — `bench.hpp`                            |
| CLI profiling (`ncu --print-summary` / `nvprof`)         | Day 5 — `ncu_collect.sh`                          |

If a row in this table has no checkbox at the end of the week, **the project is not finished**.

---

## 9. The Interview Story You'll Tell

> "In Week 5 I built FrameForge — a from-scratch CUDA preprocessor for a simulated 8-camera FSD stream. I used the APOD framework end-to-end: I assessed a naive uint8→float32 normalize kernel as memory-bound, parallelized it with a block-size sweep, optimized it with coalesced loads and pinned host memory, then deployed a multi-stream pipeline with an atomic frame-mean reduction. I measured effective bandwidth against the theoretical peak from `cudaGetDeviceProperties` and got from ~X% to ~Y% of peak. Warp-shuffle reductions and shared-memory bank tricks I'm doing next week — they're outside what the Week 5 resources actually cover."

The last sentence is the part that proves you know *what you don't know yet*. That's the version Tesla wants to hear.

Sources consulted while scoping this document:
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — Ch 1–3, §5
- [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — Ch 1–3, §9, §10
- [freeCodeCamp CUDA Course — Infatoshi/cuda-course on GitHub](https://github.com/Infatoshi/cuda-course) — Module 5 ("Writing your First Kernels") and Module 7 ("Faster Matmul") for scope boundary verification

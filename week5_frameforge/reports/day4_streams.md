# Process

- Break the larger problem into smaller chunks.
- H2D, kernel, D2H for each chunk in parallel.
- Enables pipelining and overlap of memcpy and computation.
- Should improve runtime.
- Added atomicAdd(double) for reduction. The problem is, since it is atomic, atomic writes are serialized.
- This adds significant slowdown as shown in the results.

## Results with Atomic Add

(Compare e2e of week-3 with wall of week-4)

```
pinned   + coalesced        kernel: 0.107 ms  172.7 GB/s  |  e2e: 1.546 ms
pageable + coalesced        kernel: 0.107 ms  171.8 GB/s  |  e2e: 4.199 ms
pinned   + strided          kernel: 0.422 ms  43.7 GB/s  |  e2e: 1.865 ms
pageable + strided          kernel: 0.422 ms  43.7 GB/s  |  e2e: 4.335 ms
K=1  wall: 9.619 ms  mean_|diff|=8.970e-11  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
K=2  wall: 8.938 ms  mean_|diff|=8.720e-11  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
K=4  wall: 8.574 ms  mean_|diff|=8.856e-11  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
K=8  wall: 8.433 ms  mean_|diff|=8.754e-11  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
```

## Results without Atomic Add

(Compare e2e of week-3 with wall of week-4)

```
pinned   + coalesced        kernel: 0.101 ms  182.2 GB/s  |  e2e: 1.534 ms
pageable + coalesced        kernel: 0.101 ms  181.8 GB/s  |  e2e: 3.999 ms
pinned   + strided          kernel: 0.365 ms  50.6 GB/s  |  e2e: 1.790 ms
pageable + strided          kernel: 0.354 ms  52.0 GB/s  |  e2e: 4.072 ms
K=1  wall: 2.327 ms  mean_|diff|=5.622e+02 (IGNORE)  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
K=2  wall: 2.169 ms  mean_|diff|=5.622e+02 (IGNORE)  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
K=4  wall: 2.150 ms  mean_|diff|=5.622e+02 (IGNORE)  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
K=8  wall: 2.125 ms  mean_|diff|=5.622e+02 (IGNORE)  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
```

## Why the speedup is small

The T4 has three engines that can run concurrently: an H2D copy engine, a D2H copy engine, and the SMs. With K chunks, each engine still processes its own queue serially, but the three timelines can run in parallel:

```
H2D engine:   [h1][h2]...[h8]                         (~0.30 ms total)
Kernel:           [k1][k2]...[k8]                     (~0.10 ms total)
D2H engine:           [d1][d2]...[d8]                 (~1.13 ms total)
```

The wall time is bounded below by the slowest engine's total work — here D2H, because `float32` output is 4× the size of `uint8` input. That puts the theoretical floor at about 1.13 ms versus the serial 1.53 ms baseline, a ceiling of roughly 1.35× speedup. Compute is so small relative to D2H that hiding it gains only ~9%; hiding H2D inside D2H gains another ~20%. The measured K=8 wall of 2.13 ms is above the theoretical floor primarily because the timed region includes per-call `cudaMalloc`/`cudaFree` and stream create/destroy (~0.5–0.8 ms of fixed overhead), which Day 3's benchmark did not pay. Hoisting those out of the timed region would bring K=1 down to ~1.5 ms and K=8 close to ~1.2 ms.

## Conclusion

- With atomic add, the runtime is higher due to serialization of atomic writes.
- In both cases, the wall (e2e) time reduces as K increases. The more parallelization we can do in terms of streaming, the better.
- The wall time is higher than without streaming, because the timed portion includes the initialization and allocation of device memory and streams, something that was not timed in week-3 runs.

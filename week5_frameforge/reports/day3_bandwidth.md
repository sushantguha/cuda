## Theoretical bandwidth calculation:

- Memory clock: 5001000 kHz
- Memory bus width: 256 bits
- Theoretical BW: 320.1 GB/s

## Effective bandwidth calculation:

- Data read + written: 1280 * 960 * 3 * 1 + 1280 * 960 * 3 * 4 = 18.4MB
- Time Taken (from day2): 0.117ms
- Effective bandwidth: 157GB/s

## Data from measured run:

| config | kernel time | bandwidth | e2e time |
|--------|-------------|-----------|----------|
| pinned + coalesced | 0.129 ms | 142.9 GB/s | 1.553 ms |
| pageable + coalesced | 0.113 ms | 163.4 GB/s | 4.164 ms |
| pinned + strided | 0.437 ms | 42.2 GB/s | 1.875 ms |
| pageable + strided | 0.358 ms | 51.5 GB/s | 4.619 ms |

## Analysis:

- The coalesced access pattern is much faster than strided access pattern - and demonstrates 3-4x the bandwidth. It matches our effective bandwidth calculation above.
- Pinned and Pageable memory do not affect kernel bandwidth since the data will be present in GPU global memory regardless, but pinned memory significantly reduces e2e latency. With pageable memory, CUDA must first copy the data into an internal pinned staging buffer before DMA'ing to the GPU. This means two host-side copies instead of one. This explains the ~2x higher e2e latency for pageable configs.
## Static analysis:

This kernel reads 1 byte/pixel (uint8) and writes 4 bytes/pixel (float32), for 5 bytes/element moved. Compute is 2 FLOPs/element (one subtract, one multiply). Arithmetic intensity ≈ 0.4 FLOP/byte — well below 1, placing this firmly in the memory-bound regime on the roofline. Adding more compute parallelism will not help; the bottleneck is memory bandwidth. The block-size sweep below is the Parallelize step: varying launch config to find the empirical optimum without changing the algorithm.

## The results of running the timing sweep are as follows:

| block size | median |
|------------|--------|
| 32 | 0.270 ms |
| 64 | 0.125 ms |
| 128 | 0.117 ms |
| 256 | 0.117 ms |
| 512 | 0.118 ms |
| 1024 | 0.124 ms |

## The following are given:

- Warps have a size of 32 threads
- The code ran on a T4 GPU which has a limit per SM of either 16 blocks or 32 warps.
- The scheduler always tries to fit as many warps as possible per SM.

### Block size of 32:

- We can only have 1 warp per block.
- This means we can have at most 16 blocks per SM (Limiting factor).
- Hence, only 16 warps per SM.
- This reduces the amount of context switching a single SM can do to hide memory latencies.
- This leads to poor performance.

### Block size of 64:

- We can have 2 warps per block.
- Can fit 32 warps per SM.
- This means we can have at most 16 blocks per SM.
- This leads to better performance than block size of 32.
- But still very few blocks per SM.

### Block size of 128:

- We can have (128 / 32) = 4 warps per block.
- Can fit 32 warps per SM.
- This means we can have at most 8 blocks per SM.
- This leads to better performance than block size of 64.
- Better blocks per SM.

### Block size of 256:

- We can have (256 / 32) = 8 warps per block.
- Can fit 32 warps per SM.
- This means we can have at most 4 blocks per SM.
- Full Occupancy

### Block size of 1024:

- We can have (1024 / 32) = 32 warps per block.
- Can fit 32 warps per SM.
- This means we can have at most 1 block per SM.
- Full Occupancy
- But since we only have 1 block, some flexibility is gone, worsening performance.



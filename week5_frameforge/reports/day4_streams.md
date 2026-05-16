Process:
- Break the larger problem into smaller chunks. 
- H2D, kernel, D2H for each chunk in parallel. 
- Enables pipelining and ovelap of memcpy and computation. 
- Should improve runtime. 
- Added atomicAdd(double) for reduction. The problem is, since it isa atmoic, atmoic writes are serialized. 
- This adds significant slowdown as shown in the results.

Results with Atmoic Add:
pinned   + coalesced        kernel: 0.107 ms  172.7 GB/s  |  e2e: 1.546 ms
pageable + coalesced        kernel: 0.107 ms  171.8 GB/s  |  e2e: 4.199 ms
pinned   + strided          kernel: 0.422 ms  43.7 GB/s  |  e2e: 1.865 ms
pageable + strided          kernel: 0.422 ms  43.7 GB/s  |  e2e: 4.335 ms
K=1  wall: 9.619 ms  mean_|diff|=8.970e-11  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
K=2  wall: 8.938 ms  mean_|diff|=8.720e-11  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
K=4  wall: 8.574 ms  mean_|diff|=8.856e-11  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)
K=8  wall: 8.433 ms  mean_|diff|=8.754e-11  out_max_abs=0.000e+00  out_max_rel=0.000e+00  (at i=-1)

Results without Atmoic add:
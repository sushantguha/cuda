#include <iostream>

__global__ void branch() {
    // if ((threadIdx.x / 32) % 2) {
    //     // printf("Even thread: %d\n", threadIdx.x);
    //     int a = 1;
    // } else {
    //     // printf("Odd thread: %d\n", threadIdx.x);
    //     int b = 2;
    // }

    if ((threadIdx.x % 2) == 0) {
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
    branch<<<1024, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}

// WITH WARP DIVERGENCE
//   "ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"                                                                                                                      
//   "0","2583","a","127.0.0.1","branch()","1","7","(256, 1, 1)","(1, 1, 1)","0","7.5","Command line profiler metrics","dram__bytes_read.sum","byte","2,816"                                                                                                                                             
//   "0","2583","a","127.0.0.1","branch()","1","7","(256, 1, 1)","(1, 1, 1)","0","7.5","Command line profiler metrics","dram__bytes_write.sum","byte","0"                                                                                                                                                
//   "0","2583","a","127.0.0.1","branch()","1","7","(256, 1, 1)","(1, 1, 1)","0","7.5","Command line profiler metrics","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","0.38"                                                                                                                   
//   "0","2583","a","127.0.0.1","branch()","1","7","(256, 1, 1)","(1, 1, 1)","0","7.5","Command line profiler metrics","sm__cycles_elapsed.avg","cycle","1,362.20"  

// WITHOUT WARP DIVERGENCE
// ID,Process ID,Process Name,Host Name,Kernel Name,Context,Stream,Block Size,Grid Size,Device,CC,Section Name,Metric Name,Metric Unit,Metric Value                                                                                                                                                    
//   0,3696,a,127.0.0.1,branch(),1,7,(256, 1, 1),(1, 1, 1),0,7.5,Command line profiler metrics,dram__bytes_read.sum,byte,2,816                                                                                                                                                                           
//   0,3696,a,127.0.0.1,branch(),1,7,(256, 1, 1),(1, 1, 1),0,7.5,Command line profiler metrics,dram__bytes_write.sum,byte,0                                                                                                                                                                              
//   0,3696,a,127.0.0.1,branch(),1,7,(256, 1, 1),(1, 1, 1),0,7.5,Command line profiler metrics,dram__throughput.avg.pct_of_peak_sustained_elapsed,%,0.39                                                                                                                                                 
//   0,3696,a,127.0.0.1,branch(),1,7,(256, 1, 1),(1, 1, 1),0,7.5,Command line profiler metrics,sm__cycles_elapsed.avg,cycle,1,363.20     
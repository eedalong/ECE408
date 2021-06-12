#include <cuda.h>
#include <iostream>

__global__ void TestKernel(unsigned long long* time){
    __shared__ float shared_data[1024];
    unsigned long long startTime = clock();
    int tid = threadIdx.x;
    shared_data[tid * 2] ++ ;
    unsigned long long endTime = clock();
    time = endTime - startTime;
}


int main(){
    unsigned long long time;
    unsigned long long * dtime;
    cudaMalloc((void**) &dtime, sizeof(unsigned long long));
    for(int index=0; index < 10; index++){
        TestKernel<<1, 32>>>(dtime);
        cudaMemcpy(&time, dtime, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        std::cout <<"Time: "<<(time - 14) / 32 << std::endl;
        std::cout << std::endl;
    } 
    cudaFree(dtime);
    cudaDeviceReset();
    return 0;
}
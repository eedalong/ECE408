#include <cuda.h>
#include <iostream>

/*
1. stride = 0: all threads request same value, this is where broadcasting happened
2. stride = 1: all threads request different bank
3. stride = 2: 2-way bank conflict
4. stride = 4: 4-way bank conflict
5. stride = 16: 16-way bank conflict
6. stride = 32: 32-way bank conflict

*/
__global__ void TestKernel(unsigned long long* time){
    __shared__ float shared_data[1024];
    unsigned long long startTime = clock();
    int stride = 2;
    int tid = threadIdx.x;
    shared_data[tid * stride] ++ ;
    unsigned long long endTime = clock();
    *time = endTime - startTime;
}


int main(){
    unsigned long long time;
    unsigned long long * dtime;
    cudaMalloc((void**) &dtime, sizeof(unsigned long long));
    for(int index=0; index < 10; index++){
        TestKernel<<<1, 32>>>(dtime);
        cudaMemcpy(&time, dtime, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        std::cout <<"Time: "<<(time - 14) / 32 << std::endl;
        std::cout << std::endl;
    } 
    cudaFree(dtime);
    cudaDeviceReset();
    return 0;
}
#include <cuda.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>

/*
1. stride = 0: all threads request same value, this is where broadcasting happened
2. stride = 1: all threads request different bank
3. stride = 2: 2-way bank conflict
4. stride = 4: 4-way bank conflict
5. stride = 16: 16-way bank conflict
6. stride = 32: 32-way bank conflict

*/
__global__ void TestKernel(unsigned long long* time, int stride){
    __shared__ float shared_data[1024];
    unsigned long long startTime = clock();
    int tid = threadIdx.x;
    shared_data[tid * stride] = 4;
    shared_data[tid * stride] += 4;
    shared_data[tid * stride] *= 4;
    unsigned long long endTime = clock();
    *time = (endTime - startTime);
}


int main(int argc, const char** argv){
    /*
    Bank_Conflict -s 2
    */
    if(argc != 3){
        printf("this should be used like: ./Bank_Conflict -s(stride) 2\n");
        return -1;
    }
    int stride = 0;
    for(int index = 0; index < argc; index++){
        if(strcmp(argv[index], "-s") == 0){
            stride = atoi(argv[index + 1]);
        }
    }
    std::cout << "stride set is "<<stride<<std::endl;
    unsigned long long time;
    unsigned long long * dtime;
    cudaMalloc((void**) &dtime, sizeof(unsigned long long));
    for(int index=0; index < 10; index++){
        TestKernel<<<1, 32>>>(dtime, stride);
        cudaMemcpy(&time, dtime, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        // 14 is overhead for calling clock
        std::cout <<"Time: "<<(time - 14) / 32 << std::endl;
        std::cout << std::endl;
    } 
    cudaFree(dtime);
    cudaDeviceReset();
    return 0;
}
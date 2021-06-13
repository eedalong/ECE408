// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>
#include <iostream>
#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)


int ceil(int a, int b){
    return (a + b - 1) / b;
}

__global__ void scan(float * input, float * output, float* block_sum, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

    // for each thread, we process BLOCK_SIZE * 2 elements
    __shared__ float shared_data[BLOCK_SIZE << 1];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int elementsNumPerBlock = BLOCK_SIZE << 1;
    int bid_offset = bid * elementsNumPerBlock;
    
    // each thread load 2 elements
    shared_data[2 * tid] = 0;
    shared_data[2 * tid + 1] = 0;

    if(bid_offset + 2 * tid < len)
        shared_data[2 * tid] = input[bid_offset + 2 * tid];
    if(bid_offset + 2 * tid + 1 < len)
        shared_data[2 * tid + 1] = input[bid_offset + 2 * tid + 1];
    __syncthreads();

    // up-sweep phase 
    int offset = 1;
    for(int d = elementsNumPerBlock / 2; d > 0; d >>= 1){
        if(tid < d){
            int bi = offset * 2 * (tid + 1) - 1;
            int ai = bi - offset;
            shared_data[bi] += shared_data[ai];
        }
        offset <<= 1;
        __syncthreads();
    }

    // clear last element to zero and save it to block_sum
    if(tid == 0){
        block_sum[bid] = shared_data[elementsNumPerBlock - 1];
        shared_data[elementsNumPerBlock - 1] = 0;
    }

    __syncthreads();

    // down-sweep phase
    for(int d = 1; d < elementsNumPerBlock; d <<= 1){

        if(tid < d){
            int bi = offset * 2 * (tid + 1) - 1;
            int ai = bi - offset;
            float t = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] += t;
        }
        offset >>= 1;
        __syncthreads();
    }

    __syncthreads();

    if(bid_offset + 2 * tid < len){
        output[bid_offset + 2 * tid] += shared_data[2 * tid];
    }
    if(bid_offset + 2 * tid + 1 < len){
        output[bid_offset + 2 * tid + 1] += shared_data[2 * tid + 1];
    }

}


// Grid && Block are both 1-dimensional
__global__ void uniform_add(float * input, float * block_sum, int input_len){
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int base_idx = block_idx * (BLOCK_SIZE << 1);
    // each thread process 2 elements
    if((base_idx + 2 * thread_idx) < input_len){
        input[base_idx + 2 * thread_idx] += block_sum[block_idx];
    }
    if((base_idx + 2 * thread_idx + 1) < input_len){
        input[base_idx + 2 * thread_idx + 1] += block_sum[block_idx];
    }
}


int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    float * blockSum;
    int numElements; // number of elements in the list
    int blockNum = 0;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    blockNum = ceil(numElements, BLOCK_SIZE << 1);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);
    std::cout << "The number of input elements in the input is " <<numElements<<std::endl;
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&blockSum, blockNum * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing blockSum memory.");
    //wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbCheck(cudaMemset(blockSum, 0, blockNum * sizeof(float)));
    wbTime_stop(GPU, "Clearing blockSum memory.");
    std::cout << "blockSum memory cleared"<<std::endl;
    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceOutput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 GridDim(blockNum, 1, 1);
    dim3 BlockDim(BLOCK_SIZE, 1, 1);
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    std::cout << "Performing CUDA computation"<<std::endl;
    scan<<<GridDim, BlockDim>>>(deviceInput, deviceOutput, blockSum, numElements);
    cudaDeviceSynchronize();
    std::cout << "Performing blockSum add computation"<<std::endl;
    // add block sum to each block
    // TODO Debug
    //uniform_add<<<GridDim, BlockDim>>>(deviceOutput, blockSum, numElements);
    //cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    std::cout << "Copying output memory to the CPU"<<std::endl;
    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(blockSum);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}


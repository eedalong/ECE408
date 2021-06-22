// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

// By Dalong: This implementation fails to return correct answer when input vector has more than 3M elements though idk why. I am still working on it.

#include    <wb.h>
#include <iostream>
#define BLOCK_SIZE 512 //@@ You can change this
#define ELEMENT_NUM_PER_BLOCK BLOCK_SIZE * 2
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

__global__ void pscan(float * input, float * output, float* block_sum, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

    // for each thread, we process ELEMENT_NUM_PER_BLOCK elements
    __shared__ float shared_data[ELEMENT_NUM_PER_BLOCK];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int bid_offset = bid * ELEMENT_NUM_PER_BLOCK;
    
    // each thread load 2 elements

    if((bid_offset + 2 * tid) < len)
        shared_data[2 * tid] = input[bid_offset + 2 * tid];
    else
        shared_data[2 * tid] = 0;

    if((bid_offset + 2 * tid + 1) < len)
        shared_data[2 * tid + 1] = input[bid_offset + 2 * tid + 1];
    else
        shared_data[2 * tid + 1] = 0;

    __syncthreads();

    // up-sweep phase 

    int offset = 1;
    for(int d = ELEMENT_NUM_PER_BLOCK / 2; d > 0; d /= 2){
        __syncthreads();
        if(tid < d){
            int bi = offset * 2 * (tid + 1) - 1;
            int ai = bi - offset;
            shared_data[bi] += shared_data[ai];
        }
        offset *= 2;
       
    }
    __syncthreads();

    // clear last element to zero and save it to block_sum
    if(tid == 0){
        block_sum[bid] = shared_data[ELEMENT_NUM_PER_BLOCK - 1];
        shared_data[ELEMENT_NUM_PER_BLOCK - 1] = 0;
    }

    __syncthreads();

    // down-sweep phase
    for(int d = 1; d < ELEMENT_NUM_PER_BLOCK; d *= 2){
        offset >>= 1;
        __syncthreads();
        if(tid < d){
            int bi = offset * 2 * (tid + 1) - 1;
            int ai = bi - offset;
            float t = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] += t;
        } 
        
    }

    __syncthreads();
    
    // here we get exclusive prefix sum, we add them with original data to get inclusive prefix sum
    if(bid_offset + 2 * tid < len){
        output[bid_offset + 2 * tid] = input[bid_offset + 2 * tid] + shared_data[2 * tid];
    }
    if(bid_offset + 2 * tid + 1 < len){
        output[bid_offset + 2 * tid + 1] = input[bid_offset + 2 * tid + 1] + shared_data[2 * tid + 1];
    }

}


float** g_scanBlockSums;
int maxLevel = 0;
void preallocBlockSums(unsigned int maxNumElements){
    int tempNumElements = maxNumElements;
    while(tempNumElements > 1){
        tempNumElements = ceil(tempNumElements, ELEMENT_NUM_PER_BLOCK);
        maxLevel += 1; 
    }
    maxLevel += 1;
    // allocate memory for different level of blockSum
    std::cout<<"maxLevel is "<<maxLevel<<std::endl;
    g_scanBlockSums = (float**) malloc(sizeof(float*) * maxLevel);
    tempNumElements = maxNumElements;
    int level = 0;
    while(tempNumElements > 1){
        // this is block num
        tempNumElements = ceil(tempNumElements, ELEMENT_NUM_PER_BLOCK);
        cudaMalloc((void**) &g_scanBlockSums[level], sizeof(float) * tempNumElements);
        level += 1;
    }
    // this is for the last g_scanBlockSums
    cudaMalloc((void**) &g_scanBlockSums[level], sizeof(float));
    std::cout<<"Finished preallocBlockSums"<<std::endl;

}  
void deallocBlockSums(){
    for(int level=0; level < maxLevel; level++){
        cudaFree(g_scanBlockSums[level]);
    }
    free(g_scanBlockSums);
}

// Grid && Block are both 1-dimensional
__global__ void uniform_add(float * input, float * block_sum, int input_len){
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    // blocks we process 1,2,3,4...BLOCK_NUM-1
    int base_idx = (block_idx + 1) * ELEMENT_NUM_PER_BLOCK;
    // each thread process 2 elements
    if((base_idx + 2 * thread_idx) < input_len){
        input[base_idx + 2 * thread_idx] += block_sum[block_idx];
    }
    if((base_idx + 2 * thread_idx + 1) < input_len){
        input[base_idx + 2 * thread_idx + 1] += block_sum[block_idx];
    }
}

// all array here are allocated on GPU
void scanRecursive(float* input, float* output, int elementNum, int level){
    int blockNum = ceil(elementNum, ELEMENT_NUM_PER_BLOCK);
    dim3 DimGrid(blockNum, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    pscan<<<DimGrid, DimBlock>>>(input, output, g_scanBlockSums[level], elementNum);

    // elementNum <= ELEMENT_NUM_PER_BLOCK
    // scanBlocksSum length = 1
    if(blockNum == 1){
        std::cout<< "elementNum <= ELEMENT_NUM_PER_BLOCK"<<std::endl;
        return;
    }
    // elementNum <= ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK
    // scanBlocksSum length < ELEMENT_NUM_PER_BLOCK, which can be processed by one block
    else if(blockNum <= ELEMENT_NUM_PER_BLOCK){
        std::cout<< "elementNum <= ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK"<<std::endl;
        std::cout<< "blockNum is "<<blockNum<<std::endl;
        dim3 blockSumGrid(1, 1, 1);
        std::cout<< "calculate prefix sum of g_scanBlockSums[level]"<<std::endl;
        pscan<<<blockSumGrid, DimBlock>>>(g_scanBlockSums[level], g_scanBlockSums[level], g_scanBlockSums[level + 1], blockNum);
        
    }else{
        // elementNum > ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK
        // scanBlockSum length > ELEMENT_NUM_PER_BLOCK, which need to be processed by multiple blocks
        std::cout<< "elementNum > ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK"<<std::endl;
        std::cout<< "blockNum is "<<blockNum<<std::endl;
        scanRecursive(g_scanBlockSums[level], g_scanBlockSums[level], blockNum, level + 1);
    }
    std::cout<< "add segment prefix sum to result"<<std::endl;
    // add blockSum to output.
    dim3 addGrid(blockNum-1, 1, 1);
    uniform_add<<<addGrid, DimBlock>>>(output, g_scanBlockSums[level], elementNum);
}



int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    std::cout<< "Begin to prealloc g_scanBlockSums"<<std::endl;
    preallocBlockSums(numElements);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    std::cout<< "The number of input elements in the input is " << numElements<<std::endl;
    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scanRecursive(deviceInput, deviceOutput, numElements, 0);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);
    deallocBlockSums();

    return 0;
}


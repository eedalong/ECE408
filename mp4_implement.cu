// MP 4 Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include    <wb.h>

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

__global__ void reduction_v1(float * input, float * output, int len){
     //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    __shared__ float shared_data[BLOCK_SIZE << 1];
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int idx = bx * blockDim.x + tx;
    // Load data into shared memory
    shared_data[tx] = input[idx];
    __syncthreads();
    int stride = 1;
    for(int stride = 1; stride <= blockDim.x ; stride <<= 1){
        /*
        Problem:
        1. Highly divergenet warps are very ineffiency
        2. % operator is very slow
        */
        if(tx % (stride * 2) == 0){
            shared_data[tx] += shared_data[tx + stride];
        }
        __syncthreads();
    }

    if(tx == 0) output[bx] = shared_data[0];

}

__global__ void reduction_v2(float* input, float * output, int len){
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    __shared__ float shared_data[BLOCK_SIZE << 1];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;

    // Load data into shared memory
    shared_data[tid] = input[idx];
    __syncthreads();

    // we dont filter thread id, we filter reduction index
    // i.e. for stride = 1, index should be 0, 2, 4, 6, 8...
    // for stride = 2, index should be 0, 4, 8, 12...
    // for stride = 4, index should be 0, 8, 16, 24...

    /*
    Problem: 
        1. Interleaved memory address with bank confliction.
    */
    for(unsigned int s = 1; s < blockDim.x; s <<= 1){
        int index = s * tid * 2;
        if(index < blockDim.x){
            shared_data[index] += shared_data[index + s];
        }
        __syncthreads();
    }
    // set result
    if(tid == 0) output[bid] = shared_data[tid];
}
__global__ void reduction(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    __shared__ float shared_data[BLOCK_SIZE <<1];
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int index = bx * blockDim.x + tx;
    shared_data[tx] = input[index];
    __syncthreads();
    for(unsigned int s = blockDim.x / 2; s >= 1; s >>= 1){
        if(tx < s){
            shared_data[tx] += shared_data[tx + s];
        }
        __syncthreads();

    }
    if(tx == 0) output[bx] = shared_data[0];
}


int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void **)&deviceInput, sizeof(float) * numInputElements);
    cudaMalloc((void **)&deviceOutput, sizeof(float) * numOutputElements);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, sizeof(float) * numInputElements, cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(numInputElements, BLOCK_SIZE << 1), 1, 1);
    dim3 DimBlock(BLOCK_SIZE << 1, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    reduction<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}


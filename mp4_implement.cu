// MP 4 Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define ELEMENT_NUM_PER_BLOCK BLOCK_SIZE << 1
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

__global__ void reduction(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    __shared__ float shared_data[ELEMENT_NUM_PER_BLOCK];
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int base_idx = bx * ELEMENT_NUM_PER_BLOCK;
    // each thread load 2 elements into shared memory
    if(base_idx + 2 * tx < len){
        shared_data[2 * tx] = input[base_idx + 2 * tx];
    }else{
        shared_data[2 * tx] = 0; 
    }
    if(base_idx + 2 * tx + 1 < len){
        shared_data[2 * tx + 1] = input[base_idx + 2 * tx + 1];
    }else{
        shared_data[2 * tx + 1] = 0;
    }
    __syncthreads();
    
    for(unsigned int stride = ELEMENT_NUM_PER_BLOCK / 2; stride > 0; stride /= 2){
        if(tx < stride){
            shared_data[tx] += shared_data[tx + stride];
        }
        __syncthreads();

    }
    __syncthreads();
    if(tx == 0) {
        output[bx] = shared_data[0];
    }
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

    // each block output one elements
    numOutputElements = ceil(numInputElements, ELEMENT_NUM_PER_BLOCK);
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
    dim3 DimGrid(numOutputElements, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    reduction<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * numOutputElements, cudaMemcpyDeviceToHost);

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


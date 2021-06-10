// MP 1
#include	<wb.h>

#define BLOCK_SIZE 256

int ceil(int a, int b){
    return int((a + b - 1) / b);
}
__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < length){
        out[idx] = in1[idx] + in2[idx];
    }
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1;
    float * deviceInput2;
    float * deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void**) &device_input1, sizeof(float) * inputLength);
    cudaMalloc((void**) &device_input2, sizeof(float) * inputLength);
    cudaMalloc((void**) &device_output, sizeof(float) * inputLength);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(device_input1, hostInput1, sizeof(float) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(device_input2, hostInput2, sizeof(float) * length, cudaMemcpyHostToDevice);


    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(length / BLOCK_SIZE), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    vecAdd<<<DimGrid, DimBlock>>>(device_input1, device_input2, device_output);

    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, device_output, sizeof(float) * length, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(device_input1);
    cudaFree(device_input2);
    cudaFree(device_output);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}


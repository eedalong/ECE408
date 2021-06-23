#include	<wb.h>

#define SEGMENT_LENGTH 256
#define BLOCK_SIZE 256

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < len){
        out[idx] = in1[idx] + in2[idx];
    }
}

int min(int a, int b){
    if(a < b){
        return a;
    }
    return b;
}
int ceil(int a, int b){
    return (a + b - 1) / b;
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

    cudaStream_t stream0, stream1, stream2, stream3;
    cudaStreamCreate( &stream0);
    cudaStreamCreate( &stream1);
    cudaStreamCreate( &stream2);
    cudaStreamCreate( &stream3);

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    // 1. Allocate memory on GPU
    cudaMalloc((void**) &deviceInput1, sizeof(float) * 4 * SEGMENT_LENGTH);
    cudaMalloc((void**) &deviceInput2, sizeof(float) * 4 * SEGMENT_LENGTH);
    cudaMalloc((void**) &deviceOutput, sizeof(float) * 4 * SEGMENT_LENGTH);


    dim3 DimGrid(ceil(SEGMENT_LENGTH, BLOCK_SIZE), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    // 2. do computation, Breadth First Kernel Issue
    for(int index = 0; index < inputLength; index += 4 * SEGMENT_LENGTH){
        int currentPtr1 = index;
        int currentPtr2 = currentPtr1 + SEGMENT_LENGTH;
        int currentPtr3 = currentPtr2 + SEGMENT_LENGTH;
        int currentPtr4 = currentPtr3 + SEGMENT_LENGTH;
        int length1 = 0, length2 = 0, length3 = 0, length4 = 0;
        
        // copy data
        if(currentPtr1 < inputLength){
            length1 = min(SEGMENT_LENGTH, inputLength - currentPtr1);
            cudaMemCpyAsync(&deviceInput1[0], hostInput1[currentPtr1], sizeof(float) * length1, cudaMemcpyHostToDevice, stream0);
            cudaMemCpyAsync(&deviceInput2[0], hostInput2[currentPtr1], sizeof(float) * length1, cudaMemcpyHostToDevice, stream0);
        }
        if(currentPtr2 < inputLength){
            length2 = min(SEGMENT_LENGTH, inputLength - currentPtr2);
            cudaMemCpyAsync(&deviceInput1[SEGMENT_LENGTH], hostInput1[currentPtr2], sizeof(float) * length2, cudaMemcpyHostToDevice, stream1);
            cudaMemCpyAsync(&deviceInput2[SEGMENT_LENGTH], hostInput2[currentPtr2], sizeof(float) * length2, cudaMemcpyHostToDevice, stream1);
        }
        if(currentPtr3 < inputLength){
            length3 = min(SEGMENT_LENGTH, inputLength - currentPtr3);
            cudaMemCpyAsync(&deviceInput1[SEGMENT_LENGTH * 2], hostInput1[currentPtr3], sizeof(float) * length3, cudaMemcpyHostToDevice, stream2);
            cudaMemCpyAsync(&deviceInput2[SEGMENT_LENGTH * 2], hostInput2[currentPtr3], sizeof(float) * length3, cudaMemcpyHostToDevice, stream2);
        }
        if(currentPtr4 < inputLength){
            length4 = min(SEGMENT_LENGTH, inputLength - currentPtr4);
            cudaMemCpyAsync(&deviceInput1[SEGMENT_LENGTH * 3], hostInput1[currentPtr4], sizeof(float) * length4, cudaMemcpyHostToDevice, stream3);
            cudaMemCpyAsync(&deviceInput2[SEGMENT_LENGTH * 3], hostInput2[currentPtr4], sizeof(float) * length4, cudaMemcpyHostToDevice, stream3);
        }
        // do calculation
        if(currentPtr1 < inputLength){
            vecAdd<<<DimGrid, DimBlock, stream0>>>(&deviceInput1[0], &deviceInput2[0], &deviceOutput[0], length1);
        }
        if(currentPtr2 < inputLength){
            vecAdd<<<DimGrid, DimBlock, stream1>>>(&deviceInput1[SEGMENT_LENGTH], &deviceInput2[SEGMENT_LENGTH], &deviceOutput[SEGMENT_LENGTH], length2);
        }
        if(currentPtr3 < inputLength){
            vecAdd<<<DimGrid, DimBlock, stream2>>>(&deviceInput1[SEGMENT_LENGTH * 2], &deviceInput2[SEGMENT_LENGTH * 2], &deviceOutput[SEGMENT_LENGTH * 2], length3);
        }
        if(currentPtr4 < inputLength){
            vecAdd<<<DimGrid, DimBlock, stream3>>>(&deviceInput1[SEGMENT_LENGTH * 3], &deviceInput2[SEGMENT_LENGTH * 3], &deviceOutput[SEGMENT_LENGTH * 3], length4);
        }


        // do memory copy from device to host
        if(currentPtr1 < inputLength){
            cudaMemCpyAsync(&hostOutput[currentPtr1], deviceOutput[0], sizeof(float) * length1, cudaMemcpyDeviceToHost, stream0);
        }
        if(currentPtr2 < inputLength){
            cudaMemCpyAsync(&hostOutput[currentPtr2], deviceOutput[SEGMENT_LENGTH], sizeof(float) * length2, cudaMemcpyDeviceToHost, stream1);
        }
        if(currentPtr3 < inputLength){
            cudaMemCpyAsync(&hostOutput[currentPtr3], deviceOutput[SEGMENT_LENGTH * 2], sizeof(float) * length3, cudaMemcpyDeviceToHost, stream2);
        }
        if(currentPtr4 < inputLength){
            cudaMemCpyAsync(&hostOutput[currentPtr4], deviceOutput[SEGMENT_LENGTH * 3], sizeof(float) * length4, cudaMemcpyDeviceToHost, stream3);
        }        
    }
    cudaDeviceSynchronize();

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}


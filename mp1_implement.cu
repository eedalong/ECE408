#include<stdio.h>
#include<time.h>
#include<stdlib.h>


int ceil(int a, int b){
    return (a + b - 1) / b;
}
__global__ void vector_add(float* in1, float* in2, float* out, int total_len){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < total_len){
        out[i] = in2[i] + in1[i];
    }
}

bool check_correctness(float* in1, float* in2, float* out, int N){
    int index = 0;
    for(int index = 0; index < N; index++){
        if(out[index] != (in1[index]+in2[index])){
            return false;
        }
    }
    return true;
    
}

int main(){
    srand(time(NULL));
    // make sure input length is larger than 512 but less than 65535
    int N = (rand() % (65536-512)) + 512;
    float* in1;
    float* in2;
    float* out;
    float* device_input1;
    float* device_input2;
    float* device_output;

    int index = 0;
    // allocate memory for vector
    in1 = (float*)malloc(N * sizeof(float));
    in2 = (float*)malloc(N * sizeof(float));
    out = (float*)malloc(N * sizeof(float));
    // init vector
    for(index = 0; index < N; index++){
        in1[index] = rand() * 1.0;
        in2[index] = rand() * 1.0;
    }
    // allocate device memory 
    cudaMalloc((void**)&device_input1, N * sizeof(float));
    cudaMalloc((void**)&device_input2, N * sizeof(float));
    cudaMalloc((void**)&device_output, N * sizeof(float));

    // copy vector from host to device
    cudaMemcpy(device_input1, in1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_input2, in2, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 DimGrid(ceil(N, 256), 1, 1);
    dim3 DimBlock(256, 1, 1);
    vector_add<<<DimGrid, DimBlock>>>(device_input1, device_input2, device_output, N);

    cudaDeviceSynchronize();
    cudaMemcpy(out, device_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    if(check_correctness(in1, in2, out, N)){
        printf("test passed!");
    }else{
        printf("test failed");
    }
    // free device memory
    cudaFree(device_input1);
    cudaFree(device_input2);
    cudaFree(device_output);

    // free host memory
    free(in1);
    free(in2);
    free(out);

    return 0;
}
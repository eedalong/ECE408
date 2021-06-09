#include<stdio.h>
#include<time.h>
#include<stdlib.h>

#define BLOCK_WIDTH 4

int ceil(int a, int b){
    return (a + b - 1) / b;
}
#define TILE_WIDTH 8

__global__ matrix_multiply_v1(float* A, float* B, float*C, const int M, const int N, const int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float p_value = 0.0;
    int index = 0;
    // A[row,:] * B[:, col]
    if(row < M && col < K){
        for(index = 0; index< N; index++){
            p_value += A[row * M + index] * B[index * N + col];
        }
        C[row * M + col] = p_value;
    }
}

int main(){
    srand(time(NULL));
    float* A,B,C;
    float* cudaA, cudaB, cudaC;
    int index = 0;
    int M = 64 * TILE_WIDTH;
    int N = 64 * TILE_WIDTH;
    int K = 64 * TILE_WIDTH;
    // allocate cpu memory 
    A = (float*) malloc(M * N * sizeof(float));
    B = (float*) malloc(N * K * sizeof(float));
    C = (float*) malloc(M * K * sizeof(float));

    // init cpu array
    for(index=0; index < M*N; index++){
        A[index] = rand();
    }
    for(index=0; index < N*K; index++){
        B[index] = rand();
    }


    // allocate gpu memory
    cudaMalloc((void **) &cudaA, M * N * sizeof(float));
    cudaMalloc((void **) &cudaB, N * K * sizeof(float));
    cudaMalloc((void **) &cudaC, M * K * sizeof(float));

    // copy data from host to device
    cudaMemcpy(cudaA, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // calculate result
    dim3 DimGrid(ceil(M, BLOCK_WIDTH), ceil(N, BLOCK_WIDTH));
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    matrix_multiply_v1<<<DimGrid, DimBlock>>>(cudaA, cudaB, cudaC, M, N, K);

    // move memory to host
    cudaMemcpy(C, cudaC, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // check result 







}
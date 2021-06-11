#include    <wb.h>

#define BLOCK_SIZE 8

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

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float subTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float subTileB[BLOCK_SIZE][BLOCK_SIZE];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    int step = (numAColumns + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float PValue = 0;
    for(int m = 0; m < step; m++){
        // 1. Load Matrix A Tile
        if((m * BLOCK_SIZE + tx) >= numAColumns){
            subTileA[ty][tx] = 0.0;
        }else{
            subTileA[ty][tx] = A[Row * numAColumns + m * BLOCK_SIZE + tx];
        }
        // 2. Load Matrix B Tile
        if((m * BLOCK_SIZE + ty) >= numBRows){
            subTileB[ty][tx] = 0.0;
        }else{
            subTileB[ty][tx] = B[(m * BLOCK_SIZE + ty) * numBColumns + Col];
        }
        __syncthreads();
        // 3. Calculate Multiplication
        for(int k = 0; k < BLOCK_SIZE; k++){
            PValue += subTileA[ty][k] * subTileB[k][tx];
        }
        __syncthreads();
    }
    if(Row < numCRows && Col < numCColumns){
        C[Row * numCColumns + Col] = PValue;
    }

}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    hostC = (float *) malloc(sizeof(float) * numCColumns * numCRows);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void**) &deviceA, sizeof(float) * numAColumns * numARows);
    cudaMalloc((void**) &deviceB, sizeof(float) * numBColumns * numBRows);
    cudaMalloc((void**) &deviceC, sizeof(float) * numCColumns * numCRows);

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, sizeof(float) * numAColumns * numARows, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeof(float) * numBColumns * numBRows, cudaMemcpyHostToDevice);
    

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(ceil(numCColumns, BLOCK_SIZE), ceil(numCRows, BLOCK_SIZE), 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    
    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, sizeof(float) * numCColumns * numCRows, cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}


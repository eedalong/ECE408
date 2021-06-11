#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C,
                               int numARows, int numAColumns,
                               int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {

  float Cvalue = 0;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row > numCRows || col > numCColumns) {
    return;
  }

  for (int iter = 0; iter < numAColumns; ++iter) {
    Cvalue += A[row * numAColumns + iter] * B[iter * numBColumns + col];
  }

  C[row * numCColumns + col] = Cvalue;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA;    // The A matrix
  float *hostB;    // The B matrix
  float *hostC;    // The output C matrix
  float *deviceA;  //
  float *deviceB;  //
  float *deviceC;  //
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);


  //*** Importing data and creating memory on host ***//
  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  numCRows    = numARows;
  numCColumns = numBColumns;
  hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");


  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);


  //*** Allocating GPU memory ***//
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void**) &deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void**) &deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void**) &deviceC, numCRows * numCColumns * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");


  //*** Copying input memory to the GPU ***//
  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");


  dim3 dimGrid(ceil(numCColumns/8.0), ceil(numCRows/8.0), 1);
  dim3 dimBlock(8, 8, 1);


  //*** Performing CUDA computation ***//
  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiply<<<dimGrid, dimBlock>>>(
    deviceA, deviceB, deviceC,
    numARows, numAColumns,
    numBRows, numBColumns,
    numCRows, numCColumns
  );
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");


  //*** Copying output memory to the CPU ***//
  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");


  //*** Freeing GPU Memory ***//
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");


  //*** Check Solution ***//
  wbSolution(args, hostC, numCRows, numCColumns);


  //*** Freeing CPU Memory ***//
  free(hostA);
  free(hostB);
  free(hostC);


  //*** Exit ***//
  return 0;
}


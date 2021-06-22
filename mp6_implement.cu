#include    <wb.h>
#include    <iostream>

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

#define TILE_SIZE 16
#define BLOCK_SIZE TILE_SIZE + Mask_width - 1
__constant__ float deviceKernel[Mask_width][Mask_width];


int ceil(int a, int b){
    return (a + b - 1) / b;
}

//@@ INSERT CODE HERE
/*
 |---------------> tx
 |
 |
 |
 |
 V ty
*/

__global__ void conv2d(float* inputImage, float* outputImage, int current_channel, int imageHeight, int imageWidth, int imageChannel){
    __shared__ float input_tile[TILE_SIZE + Mask_width - 1][TILE_SIZE + Mask_width - 1];
    int ty = threadIdx.y; int tx = threadIdx.x;
    int by = blockIdx.y;  int bx = blockIdx.x;
    
    int row_out = by * TILE_SIZE + ty;
    int col_out = bx * TILE_SIZE + tx;
    
    int row_in = row_out - Mask_radius;
    int col_in = col_out - Mask_radius;

    int row_index = 0;
    int col_index = 0;


    // load data
    if((row_in >= 0) && (row_in < imageHeight) && (col_in >= 0) && (col_in < imageWidth)){
        input_tile[ty][tx] = inputImage[(row_in * imageWidth + col_in) * imageChannel + current_channel];
    }else{
        input_tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    // compute
    float output = 0.0f;
    if(tx < TILE_SIZE && ty < TILE_SIZE){
        for(row_index = 0; row_index < Mask_width; row_index++){
            for(col_index = 0; col_index < Mask_width; col_index++){
                output += deviceKernel[row_index][col_index] * input_tile[row_index+ty][col_index+tx];
            }
        }
        // set output
        if(row_out < imageHeight && col_out < imageWidth){
            outputImage[(row_out * imageWidth + col_out) * imageChannel + current_channel] = output;
        }
    }
}


int main(int argc, char* argv[]) {
    wbArg_t arg;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;

    arg = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(arg, 0);
    inputMaskFile = wbArg_getInputFile(arg, 1);

    inputImage = wbPPM_import(inputImageFile);

    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);


    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);


    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    // use constant memory for deviceKernel
    cudaMemcpyToSymbol(deviceKernel,
                       hostMaskData,
                       Mask_width * Mask_width * sizeof(float),
                       0,
                       cudaMemcpyHostToDevice
                      );
     
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    dim3 DimGrid(ceil(imageWidth, TILE_SIZE), ceil(imageHeight, TILE_SIZE), 1);
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    std::cout <<"begin to do GPU computation"<<std::endl;
    //@@ INSERT CODE HERE
    for(int current_channel = 0; current_channel < imageChannels; current_channel ++){
        conv2d<<<DimGrid, DimBlock>>>(deviceInputImageData, deviceOutputImageData, current_channel, imageHeight, imageWidth, imageChannels);
    }
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Doing the computation on the GPU");

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
        
    std::cout<<"check mask "<<std::endl;
    for(int row = 0; row < 5; row ++){
        for(int col = 0; col < 5; col ++){
            std::cout<<hostMaskData[row * Mask_width + col]<<", ";
        }
        std::cout<<endl;
    }

    std::cout<<"check output "<<std::endl;
    for(int row = 0; row < 5; row ++){
        for(int col = 0; col < 5; col ++){
            std::cout<<hostOutputImageData[(row * imageWidth + col) * imageChannels + 0]<<", ";
        }
        std::cout<<endl;
    }
    
    wbSolution(arg, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

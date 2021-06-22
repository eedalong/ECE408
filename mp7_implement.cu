// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_WIDTH 32
#define CHANNEL 3

int ceil(int a, int b){
    return (a + b - 1) / b;
}
//@@ insert code here

// pscan
__global__ void cal_cdf(int * inputHist，int * cdf) {

    /*
        calculate cdf 
    */
    __shared__ int shared_data[HISTOGRAM_LENGTH];

    int tid = threadIdx.x;    

    // each thread load 1 element
    if(tid < HISTOGRAM_LENGTH){
        shared_data[tid] = inputHist[tid];
    }
    __syncthreads();

    // up-sweep phase 

    int offset = 1;
    for(int d = HISTOGRAM_LENGTH / 2; d > 0; d /= 2){
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
        shared_data[HISTOGRAM_LENGTH - 1] = 0;
    }

    __syncthreads();

    // down-sweep phase
    for(int d = 1; d < HISTOGRAM_LENGTH / 2; d *= 2){
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
    if(tid < HISTOGRAM_LENGTH){
        cdf[tid] = inputHist[tid] + shared_data[tid];
    }
}

__global__ void histogram_equalization(float * deviceInputImage, float* deviceOutputImage, int* cdf, int width, int height){
    
    //
    int by = blockIdx.y;
    int bx = blockIdx.x;
    //
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // coordinate
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    int channel = blockIdx.z;

    if(row < height && col < width){
        int val = (unsigned char)deviceInputImage[(row * width + col) * CHANNEL + channel];
        deviceOutputImage[(row * width + col) * CHANNEL + channel] = (unsigned char)(255.0*(cdf[val] - cdf[0])/(cdf[HISTOGRAM_LENGTH - 1] - cdf[0])) / 255.0;
    }
}


// calculate hist
__global__ void hist(unsigned char* inputImage, int length, int* hist_output){
    __shared__ unsigned int hist[HISTOGRAM_LENGTH];
    // init 
    if(threadIdx.x < HISTOGRAM_LENGTH){
        hist[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // accumulate 
    int pixel = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while(pixel < length){
        atmicAdd(&(hist[inputImage[pixel]]), 1);
        pixel += stride;
    }
    __syncthreads();

    // copy output to global memory
    if(threadIdx.x < 256){
        atomicAdd(&(hist_output[threadIdx.x]), 1);
    }
}

//
__global__ void cast_and_convert(float* inputImage, unsigned char* outputImage, int height, unsigned int width){
    // get block corrdination 
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // get thead coordination
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // pixel = blockId * BlockSize + threadId
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // cast data type
    if(row < height && col < width){
        float res = 0.0;
        res += 0.21 * (unsigned char)inputImage[(row * height + col) * CHANNEL];
        res += 0.71 * (unsigned char)inputImage[(row * height + col) * CHANNEL + 1];
        res += 0.07 * (unsigned char)inputImage[(row * height + col) * CHANNEL + 2];
        outputImage[row * width + col] = res;
    }

}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here

    float * deviceInputImageData
    unsigned char * deviceInputImageDataGray;
    float * deviceOutputImageData; 
    unsigned int *  deviceHist;
    unsigned int *  deviceCDF;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbPPM_import(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    // -3. initialize hostInputImageData and hostOutputImageData
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
    
    // -2. allocate memmory on GPU
    cudaMalloc((void **)&deviceInputImageData, sizeof(float) * imageHeight * imageWidth * imageChannels);
    cudaMalloc((void **)&deviceInputImageDataGray, sizeof(unsigned char) * imageHeight * imageWidth);
    cudaMalloc((void **)&deviceOutputImageData, sizeof(float) * imageHeight * imageWidth * imageChannels);
    cudaMalloc((void **)&deviceHist, sizeof(unsigned int) * HISTOGRAM_LENGTH);
    cudaMalloc((void **)&cdf, sizeof(unsigned int) * HISTOGRAM_LENGTH);

    // -1. copy memory to GPU
    cudaMemcpy(deviceInputImageData, hostInputImageData, sizeof(float) * imageHeight * imageWidth * imageChannels, cudaMemcpyHostToDevice);

    // 0. do GPU computation
    dim3 DimGrid1(ceil(imageWidth, BLOCK_WIDTH), ceil(imageHeight, BLOCK_WIDTH), 1);
    dim3 DimBlock1(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // 1. cast float to unsigned char
    cast_and_convert<<<DimGrid1, DimBlock1>>>(deviceInputImageData, deviceInputImageDataGray);

    // 2. calculate hist 
    dim3 DimGrid2(ceil(imageHeight * imageWidth, BLOCK_WIDTH * BLOCK_WIDTH), 1, 1);
    dim3 DimBlock2(BLOCK_WIDTH * BLOCK_WIDTH, 1, 1)
    hist<<<DimGrid2, DimBlock2>>>(deviceInputImageDataGray, deviceHist);

    // 3. calculate cdf
    cal_cdf<<<HISTOGRAM_LENGTH>>>(deviceHist, deviceCDF);

    // 4. histogram equalization

    dim3 DimGrid3(ceil(imageWidth, BLOCK_WIDTH), ceil(imageHeight, BLOCK_WIDTH), 3);
    dim3 DimBlock3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    histogram_equalization<<<DimGrid3, DimBlock3>>>(deviceInputImageDataFloat, deviceOutputImageData, deviceCDF, imageHeight, imageWidth);

    // 5. memcpy output to host
    cudaMemcpy(hostOutputImageData, deviceOutputImageData, sizeof(float) * imageWidth * imageHeight * imageChannels);

    // 6. validate the solution
    wbSolution(args, outputImage);

    //@@ insert code here

    // 7. free GPU memory
    cudaFree(deviceCDF);
    cudaFree(deviceHist);
    cudaFree(deviceInputImageData);
    cudaFree(deviceInputImageDataFloat);
    cudaFree(deviceOutputImageData);
   
    // 9. delete image, free cpu memory
    wbImage_delete(inputImage);
    wbImage_delete(outputImage);
    

    return 0;
}


#include <wb.h>
#include <stdlib.h>
#include <time.h>

#include <string>
#include <iostream>

#define Mask_width  5
#define Mask_radius Mask_width/2

void generate_tensor(float* imageData, int imageHeight, int imageWidth, int imageChannels){
    int index = 0;
    for(int row = 0; row < imageHeight; row ++){
        for(int col = 0; col < imageWidth; col ++){
            for(int channel = 0; channel < imageChannels; channel ++){
                imageData[index++] = (rand()%255) / 255.0;
            }
        }
    }
}

void convNd(float* imageData, float* mask_data, float* outputImage, int imageHeight, int imageWidth, int imageChannels){
    for(int row = 0; row < imageHeight; row++){
        for(int col = 0; col < imageWidth; col++){
            for(int channel = 0; channel < imageChannels; channel++){
                float output = 0;
                for(int i = -Mask_radius; i <= Mask_radius; i++){
                    for(int j = -Mask_radius; j <= Mask_radius; j++){
                        if(row + i >= 0 && row + i < imageHeight && col + j >= 0 && col + j < imageWidth){
                            output += mask_data[(i + Mask_radius) * Mask_width + (j + Mask_radius)] * imageData[(row * imageWidth + col) * imageChannels + channel];
                        }
                    }
                }
                outputImage[(row * imageWidth + col) * imageChannels + channel] = output;
            }
        }
    }
}

int main(int argc, char** argv){
    // MP6_Dataset -d [directory] -image_height [image height] -image_width [image width] -mask_width [mask width]
    if(argc != 7){
        printf("use example: MP6_Dataset -d [directory] -image_height [image height] -image_width [image width]\n");
        exit(-1);
    }
    srand(time(NULL));
    int imageHeight, imageWidth;
    int imageChannels = 3;
    std::string directory;
    int arg_index = 0;
    while (/* condition */ arg_index < argc)
    {
        /* code */
        if(strcmp(argv[arg_index], "-d") == 0){
            arg_index += 1;
            directory = std::string(argv[arg_index]);
        }
        if(strcmp(argv[arg_index], "-image_height") == 0){
            arg_index += 1;
            imageHeight = atoi(argv[arg_index]);
        }
        if(strcmp(argv[arg_index], "-image_width") == 0){
            arg_index += 1;
            imageWidth = atoi(argv[arg_index]);
        }
        arg_index += 1;
    }

    wbLog(TRACE, "data directory is: ", directory);
    wbLog(TRACE, "image shape: ", imageHeight, " x ", imageWidth, " x ", imageChannels);

    wbImage_t inputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbImage_t outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    
    float* inputImageData = wbImage_getData(inputImage);
    float* outputImageData = wbImage_getData(outputImage);
    float* maskData = (float*) malloc(sizeof(float) * Mask_width * Mask_width);

    // generate random tensor
    generate_tensor(inputImageData, imageHeight, imageWidth, imageChannels);
    generate_tensor(maskData, Mask_width, Mask_width, 1);

    convNd(inputImageData, maskData, outputImageData, imageHeight, imageWidth, imageChannels);

    std::string input_file = directory + string("/input0.ppm");
    std::string input_mask = directory + string("/input1.raw");
    std::string output_file = directory + string("/output.ppm");

    wbPPM_export(input_file.c_str(), inputImage);
    wbPPM_export(output_file.c_str(), outputImage);
    wbExport(input_mask.c_str(), maskData, Mask_width, Mask_width);

}


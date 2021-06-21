#include <wb.h>
#include <stdlib.h>
#include <time.h>

#include <string>
#include <iostream>

#define Mask_width  5
#define Mask_radius Mask_width/2

static inline void imageSetPixel(wbImage_t img, int row, int col, int c, float val) {
    float * data = wbImage_getData(img);
    int channels = wbImage_getChannels(img);
    int pitch = wbImage_getPitch(img);

    data[(row * wbImage_getWidth(img) + col) * channels + c] = val;

    return ;
}

static inline float imageGetPixel(wbImage_t img, int row, int col, int c) {
    float * data = wbImage_getData(img);
    int channels = wbImage_getChannels(img);
    int pitch = wbImage_getPitch(img);

    return data[(row * wbImage_getWidth(img) + col) * channels + c];
}

void generate_image(float* imageData, int imageHeight, int imageWidth, int imageChannels){
    int index = 0;
    for(int row = 0; row < imageHeight; row ++){
        for(int col = 0; col < imageWidth; col ++){
            for(int channel = 0; channel < imageChannels; channel ++){
                imageData[index++] = (rand()%255) / 255.0;
            }
        }
    }
    
}
void initialize_image(wbImage_t image){
    for(int row = 0; row < wbImage_getHeight(image); row ++){
        for(int col = 0; col < wbImage_getWidth(image); col ++){
            for(int channel = 0; channel < wbImage_getChannels(image); channel ++){
                imageSetPixel(image, row, col, channel, rand()%255 / 255.0);
            }
        }
    }
}

void generate_mask(float* maskData){
    /*
    0.,0.,0.077,0.,0.
    0.,0.077,0.077,0.077,0.
    0.077,0.077,0.077,0.077,0.077
    0.,0.077,0.077,0.077,0.
    0.,0.,0.077,0.,0.
    */
   generate_image(maskData, Mask_width, Mask_width, 1);
   float sum = 0.0f;
   for(int index = 0; index < Mask_width * Mask_width; index++){
       sum += maskData[index];
   }
   for(int index = 0; index < Mask_width * Mask_width; index++){
       maskData[index] = 1.0f/25;
   }
}

void convNd(wbImage_t inputImage, float* mask_data, wbImage_t& outImage){

    for(int channel = 0; channel < wbImage_getChannels(inputImage); channel ++){
        for(int row = 0; row < wbImage_getHeight(inputImage); row ++){
            for(int col = 0; col < wbImage_getWidth(inputImage);col++){
                float output = 0;
                for(int i = -Mask_radius; i <= Mask_radius; i++){
                    for(int j = -Mask_radius; j <= Mask_radius; j++){
                        if(row + i >= 0 && row + i < wbImage_getHeight(inputImage) && col + j >= 0 && col + j < wbImage_getWidth(inputImage)){
                            output += mask_data[(i + Mask_radius) * Mask_width + (j + Mask_radius)] * imageGetPixel(inputImage, row + i, col + j, channel);

                        }
                    }
                }
                
                imageSetPixel(outImage, row, col, channel, output);
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
    initialize_image(inputImage);
    generate_mask(maskData);
    std::cout<<"check inputImage "<<std::endl;
    for(int row = 0; row < 5; row ++){
        for(int col = 0; col < 5; col ++){
            std::cout<<imageGetPixel(inputImage, row, col, 0)<<", ";
        }
        std::cout<<endl;
    }
    std::cout<<"check mask "<<std::endl;
    for(int row = 0; row < 5; row ++){
        for(int col = 0; col < 5; col ++){
            std::cout<<maskData[row * Mask_width + col]<<", ";
        }
        std::cout<<endl;
    }
    


    convNd(inputImage, maskData, outputImage);

    std::cout<<"check outputImage "<<std::endl;
    for(int row = 0; row < 5; row ++){
        for(int col = 0; col < 5; col ++){
            std::cout<<imageGetPixel(outputImage, row, col, 0)<<", ";
        }
        std::cout<<endl;
    }


    std::string input_file = directory + string("/input0.ppm");
    std::string input_mask = directory + string("/input1.raw");
    std::string output_file = directory + string("/output.ppm");

    wbPPM_export(input_file.c_str(), inputImage);
    wbPPM_export(output_file.c_str(), outputImage);
    wbExport(input_mask.c_str(), maskData, Mask_width, Mask_width);

}


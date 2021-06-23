#include <wb.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#define HISTOGRAM_LENGTH 256
int globalHist[HISTOGRAM_LENGTH];
float globalCDF[HISTOGRAM_LENGTH];

void initGlobalHist(){
    for(int index = 0; index < HISTOGRAM_LENGTH; index++){
        globalHist[index] = 0;
        globalCDF[index] = 0;
    }
}
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

void initializeImage(wbImage_t image){
    for(int row = 0; row < wbImage_getHeight(image); row ++){
        for(int col = 0; col < wbImage_getWidth(image); col ++){
            for(int channel = 0; channel < wbImage_getChannels(image); channel ++){
                imageSetPixel(image, row, col, channel, rand()%255 / 255.0);
            }
        }
    }
}

void cast_and_convert_and_hist(wbImage_t inputImage){
    //wbImage_t grayImage = wbImage_new(wbImage_getWidth(inputImage), wbImage_getHeight(inputImage), 1);
    for(int row = 0; row < wbImage_getHeight(inputImage); row ++){
        for(int col = 0; col < wbImage_getWidth(inputImage); col ++){
            
            unsigned char r = (unsigned char) 255 * imageGetPixel(inputImage, row, col, 0);
            unsigned char g = (unsigned char) 255 * imageGetPixel(inputImage, row, col, 1);
            unsigned char b = (unsigned char) 255 * imageGetPixel(inputImage, row, col, 2);
            unsigned char grayValue = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
            //imageSetPixel(grayImage, row, col, 0, (float)grayValue);
            globalHist[grayValue] += 1;
        }
    }
}

unsigned char clamp(unsigned char val, unsigned char min, unsigned char max){
    if(val < min){
        return min;
    }
    if(val > max){
        return max;
    }
    return val;
}

unsigned char correct_color(unsigned char val){

    //std::cout<<"val = "<<int(val)<<" globalCDF[val] = "<<globalCDF[val] <<" globalCDF[0]"<<globalCDF[0]<<std::endl;

    return (unsigned char)(255*(globalCDF[val] - globalCDF[0])/(1 - globalCDF[0]));
}

void calCDF(int height, int width){
    globalCDF[0] = globalHist[0] * 1.0/ (height * width);
    for(int index = 1; index < HISTOGRAM_LENGTH; index++){
        globalCDF[index] = globalHist[index] * 1.0/ (height * width) + globalCDF[index-1];
    }
}

void adjust(wbImage_t inputImage, wbImage_t outputImage){
    for(int row = 0; row < wbImage_getHeight(inputImage); row ++){
        for(int col = 0; col < wbImage_getWidth(inputImage); col ++){
            for(int channel = 0; channel < wbImage_getChannels(inputImage); channel++){
                unsigned char pixel_value = (unsigned char)(255 * imageGetPixel(inputImage, row, col, channel));
                unsigned char corrected_value = correct_color(pixel_value);
               
                imageSetPixel(outputImage, row, col, channel, corrected_value / 255.0);
            }
        }
    }
}



int main(int argc, char** argv){
    // MP7_Dataset -d [directory] -image_width [image width] -image_height [image_height]
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

    // initialize image
    initializeImage(inputImage);
    initGlobalHist();

    // histogram equalization
    cast_and_convert_and_hist(inputImage);
    calCDF(wbImage_getHeight(inputImage), wbImage_getWidth(inputImage));
    std::cout<<"check globalCDF"<<std::endl;
    for(int index = 0; index < HISTOGRAM_LENGTH; index++){
        std::cout<< globalCDF[index]<<", ";
    }
    std::cout<<std::endl;
    adjust(inputImage, outputImage);

    // export image
    std::string input_file = directory + string("/input.ppm");
    std::string output_file = directory + string("/output.ppm");

    // export image
    wbPPM_export(input_file.c_str(), inputImage);
    wbPPM_export(output_file.c_str(), outputImage);

}
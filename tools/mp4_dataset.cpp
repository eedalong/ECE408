#include <wb.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <time.h>


void createArray(float* inputArray, int array_length){
    for(int index = 0; index < array_length; index++){
        inputArray[index] = rand() % 10;
    }
}

void reductionSum(float* inputArray, float* outputArray, int array_length){
    for(int index = 0; index < array_length; index++){
        outputArray[0] += inputArray[index];
    }

}
int main(int argc, char** argv){
    // MP4_Dataset -d [diretory] -size [length]
    srand(time(NULL));
    if(argc != 5){
        printf("this should be used like: MP4_Dataset -d [diretory] -size [length of input array]");
    }
    int index = 1;
    int array_size = 0;
    float * input_array;
    float * output_array;
    string directory;
    while(index < argc){
        if(strcmp(argv[index], "-size") == 0){
            index += 1;
            array_size = atoi(argv[index]);
        }
        else if(strcmp(argv[index], "-d") == 0){
            index += 1;
            directory = string(argv[index]);
        }
        index += 1;
    }
    wbLog(TRACE, "array size is  ", array_size);
    // allocate memory
    input_array = (float *)malloc(sizeof(float) * array_size);
    output_array = (float *)malloc(sizeof(float));

    // create array
    createArray(input_array, array_size);
    reductionSum(input_array, output_array, array_size);

    // set input and output path
    string inputPath = directory + string("/input.raw");
    string outputPath = directory + string("/output.raw");

    wbLog(TRACE, "input file is ", inputPath.c_str());
    wbLog(TRACE, "output file is ", outputPath.c_str());

    wbExport(inputPath.c_str(), input_array, array_size);
    wbExport(outputPath.c_str(), output_array, 1);

    return 0;

}
#include <wb.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <time.h>


void createArray(float* inputArray, int array_length){
    for(int index = 0; index < array_length; index++){
        inputArray[index] = rand() % 1131;
    }
}

void prefixSum(float* inputArray, float* outputArray, int array_length){
    outputArray[0] = inputArray[0];
    for(int index = 1; index < array_length; index++){
        outputArray[index] = outputArray[index - 1] + inputArray[index];
    }

}
int main(int argc, char** argv){
    // MP5_Dataset -d [diretory] -size [length]
    srand(time(NULL));
    if(argc != 5){
        printf("this should be used like: MP5_Dataset -d [diretory] -size [length of input array]");
    }
    int index = 1;
    int array_size = 0;
    float * input_array;
    float * output_array;
    string directory;
    while(index < argc){
        if(strcmp(argv[index], "-size")){
            index += 1;
            array_size = atoi(argv[index]);
        }
        else if(strcmp(argv[index], "-d")){
            index += 1;
            directory = string(argv[index]);
        }
        index += 1;
    }
    // allocate memory
    input_array = (float *)malloc(sizeof(float) * array_size);
    output_array = (float *)malloc(sizeof(float) * array_size);

    // create array
    createArray(input_array, array_size);
    prefixSum(input_array, output_array, array_size);

    // set input and output path
    string inputPath = directory + string("/input.raw");
    string outputPath = directory + string("./output.raw");

    wbExport(inputPath.c_str(), input_array, array_size);
    wbExport(outputPath.c_str(), output_array, array_size);

    return 0;

}
#include <wb.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <string>

void initVec(float* input, int length){
    for(int index = 0; index < length; index++){
        input[index] = rand() % 255;
    }
}

void vecAdd(float* input1, float* input2, float* output, int length){
    for(int index = 0; index < length; index++){
        output[index] = input1[index] + input2[index];
    }
}
int main(int argc, char** argv){
    // MP1_Dataset -d [directory] -vectorLength [vector length]

    if(argc != 5){
        std::cout<< "this should be used like: "<<"MP1_Dataset -d [directory] -vectorLength [vector length]"<<std::endl;
        exit(-1);
    }
    int argIndex = 0;
    int vectorLength = 0;
    std:;string directory;
    while(argIndex < argc){
        if(strcmp(argv[argIndex], "-d") == 0){
            argIndex += 1;
            directory = std::string(argv[argIndex]);
        }
        if(strcmp(argv[argIndex], "-vectorLength") == 0){
            argIndex += 1;
            vectorLength = atoi(argv[argIndex]);
        }
        argIndex += 1;
    }

    float* inputVec1 = (float*) malloc(sizeof(float) * vectorLength);
    float* inputVec2 = (float*) malloc(sizeof(float) * vectorLength);
    float* outputVec = (float*) malloc(sizeof(float) * vectorLength);

    initVec(inputVec1, vectorLength);
    initVec(inputVec2, vectorLength);
    vecAdd(inputVec1, inputVec2, outputVec, vectorLength);


    // void wbExport(const char * file, wbReal_t * data, int rows);
    
    std::string inputFile1 = directory + string("/input0.ppm");
    std::string inputFile2 = directory + string("/input1.ppm");
    std::string outputFile = directory + string("/output.ppm");

    wbExport(inputFile1.c_str(), inputVec1, vectorLength);
    wbExport(inputFile2.c_str(), inputVec2, vectorLength);
    wbExport(outputFile.c_str(), outputVec, vectorLength);

    free(inputVec1);
    free(inputVec2);
    free(outputVec);



}
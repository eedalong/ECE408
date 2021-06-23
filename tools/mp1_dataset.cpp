#include <wb.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <time.h>

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
    // MP1_Dataset -d [directory] -vector_length [vector length]

    if(argc != 5){
        std::cout<< "this should be used like: "<<"MP1_Dataset -d [directory] -vector_length [vector length]"<<std::endl;
        exit(-1);
    }
    srand(time(NULL));
    int argIndex = 0;
    int vectorLength = 0;
    std:;string directory;
    while(argIndex < argc){
        if(strcmp(argv[argIndex], "-d") == 0){
            argIndex += 1;
            directory = std::string(argv[argIndex]);
        }
        if(strcmp(argv[argIndex], "-vector_length") == 0){
            argIndex += 1;
            vectorLength = atoi(argv[argIndex]);
        }
        argIndex += 1;
    }
    wbLog(TRACE, "vec length", vectorLength);

    float* inputVec1 = (float*) malloc(sizeof(float) * vectorLength);
    float* inputVec2 = (float*) malloc(sizeof(float) * vectorLength);
    float* outputVec = (float*) malloc(sizeof(float) * vectorLength);

    initVec(inputVec1, vectorLength);
    initVec(inputVec2, vectorLength);
    vecAdd(inputVec1, inputVec2, outputVec, vectorLength);

    std::cout<<"check Vec"<<std::endl;
    for(int index = 0; index < 10; index ++){
        std::cout<<"( "<<inputVec1[index]<<", "<<inputVec2[index]<<", "<<outputVec[index]<<" )"<<std::endl;
    }


    // void wbExport(const char * file, wbReal_t * data, int rows);
    
    std::string inputFile1 = directory + string("/input0.raw");
    std::string inputFile2 = directory + string("/input1.raw");
    std::string outputFile = directory + string("/output.raw");

    wbLog(TRACE, "input file 1 path: ", inputFile1);
    wbLog(TRACE, "input file 2 path: ", inputFile2);
    wbLog(TRACE, "output file path: ", outputFile);


    wbExport(inputFile1.c_str(), inputVec1, vectorLength);
    wbExport(inputFile2.c_str(), inputVec2, vectorLength);
    wbExport(outputFile.c_str(), outputVec, vectorLength);

    free(inputVec1);
    free(inputVec2);
    free(outputVec);



}
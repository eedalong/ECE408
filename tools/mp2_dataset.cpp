#include <wb.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <time.h>
using namespace std;

void create_mat(float* mat, int row, int col){
    for(int index = 0; index < row * col; index++){
        mat[index] = float((rand() % 1337)) / 1111.0;
    }
}

void mat_multiply(float* A, float* B, float* C, int rowa, int cola, int rowb, int colb){
    for(int index = 0; index < rowa * rowb; index++){
        int row = index / colb;
        int col = index % colb;
        float Pvalue = 0;
        for(int i = 0; i < cola; i++){
            Pvalue += A[row * cola + i] * B[i * colb + col];
        }
        C[index] = Pvalue;
    }
}
int main(int argc, const char** argv) {
    if(argc != 11){
        printf("input arg count is %d\n", argc);
        printf("tool should be used as MP2_Dataset -d [directory/to/your/path/] -ra [row count of matrix a] -ca [column count of matrix a] -rb [row count of matrix b] -cb [column count of matrix b]\n");
        return -1;
    }
    srand(time(NULL));
    char* directory = nullptr;
    int cola = 0, colb = 0;
    int rowa = 0, rowb = 0;
    int index = 0;
    while(index < argc){
        if(strcmp(argv[index], "-d") == 0){
            index += 1;
            directory = (char*) malloc(strlen(argv[index]));
            strcpy(directory, argv[index]);
            
        }
        else if(strcmp(argv[index], "-ra") == 0){
            index += 1;
            rowa = atoi(argv[index]);
        }
        else if(strcmp(argv[index], "-rb") == 0){
            index += 1;
            rowb = atoi(argv[index]);
        }
        else if(strcmp(argv[index], "-ca") == 0){
            index += 1;
            cola = atoi(argv[index]);
        }
        else if(strcmp(argv[index], "-cb") == 0){
            index += 1;
            colb = atoi(argv[index]);
        }
        index += 1;

    }
    // check parameters
    if(directory == nullptr){
        printf("directory path is not correctly set\n");
        return -1;
    }
    if(cola == 0 || colb == 0 ||rowa == 0 || rowb == 0){
        printf("matrix shape is not correctly set\n");
        return -1;
    }
    wbAssert(cola == rowb);

    // allocate memory
    float* hostA;
    float* hostB;
    float* output;
    hostA = (float*) malloc(sizeof(float) * rowa * cola);
    hostB = (float*) malloc(sizeof(float) * rowb * colb);
    output = (float*) malloc(sizeof(float) * rowa * colb);

    // generate matrix
    create_mat(hostA, rowa, cola);
    create_mat(hostB, rowb, colb);

    // calculate output
    mat_multiply(hostA, hostB, output, rowa, cola, rowb, colb);

    // export
    wbLog(TRACE, "data directory is  ", directory); 
    char* input_file1 = "";
    strcat(input_file1, directory);
    strcat(input_file1, "/input0.raw");
    char* input_file2 = "";
    strcat(input_file2, directory);
    strcat(input_file2, "/input1.raw");
    char* output_file = "";
    strcat(output_file, directory);
    strcat(output_file, "/input1.raw");

    wbLog(TRACE, "input file1 is  ", input_file1);
    wbLog(TRACE, "input file2 is  ", input_file1);
    wbLog(TRACE, "output file is  ", output_file);

    wbExport(input_file1, hostA, rowa, cola);
    wbExport(input_file2, hostB, rowb, colb);
    wbExport(output_file, output, rowa, cola);

    
    return 0;


}
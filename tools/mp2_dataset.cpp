#include <wb.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <time.h>
using namespace std;

void create_mat(float* mat, int row, int col){
    for(int index = 0; index < row * col; index++){
        //mat[index] = float((rand() % 1337)) / 1111.0;
        mat[index] = rand() % 17;
    }
}

void mat_multiply(float* A, float* B, float* C, int rowa, int cola, int rowb, int colb){
    for(int i = 0; i < rowa; i++){
        for(int j = 0; j < colb; j ++){
            float Pvalue = 0;
            for(int k = 0; k < cola; k++){
                Pvalue += A[i * cola + k] * B[k  * colb + j];
            }
            C[i * colb + j] = Pvalue;
        }
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
    int index = 1;
    while(index < argc){
        if(strcmp(argv[index], "-d") == 0){
            index += 1;
            directory = (char*) malloc(strlen(argv[index]));
            strcpy(directory, argv[index]);
            wbLog(TRACE, "data directory is  ", directory); 
            
            
        }
        else if(strcmp(argv[index], "-ra") == 0){
            index += 1;
            rowa = atoi(argv[index]);
            wbLog(TRACE, "rowa is  ", rowa); 
        }
        else if(strcmp(argv[index], "-rb") == 0){
            index += 1;
            rowb = atoi(argv[index]);
            wbLog(TRACE, "rowb is  ", rowb); 
        }
        else if(strcmp(argv[index], "-ca") == 0){
            index += 1;
            cola = atoi(argv[index]);
            wbLog(TRACE, "cola is  ", cola); 
        }
        else if(strcmp(argv[index], "-cb") == 0){
            index += 1;
            colb = atoi(argv[index]);
            wbLog(TRACE, "colb is  ", colb); 
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

 
    
    // allocate memory
    float* hostA;
    float* hostB;
    float* output;
    wbLog(TRACE, "hostA size is", rowa, "X", cola);
    wbLog(TRACE, "hostB size is", rowb, "X", colb);
    wbLog(TRACE, "hostB size is", rowa, "X", colb);
    

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
    string input_file1 = string(directory) + string("/input0.raw");
    string input_file2 = string(directory) + string("/input1.raw");
    string output_file = string(directory) + string("/output.raw");


    wbLog(TRACE, "input file1 is  ", input_file1.c_str());
    wbLog(TRACE, "input file2 is  ", input_file2.c_str());
    wbLog(TRACE, "output file is  ", output_file.c_str());

    wbExport(input_file1.c_str(), hostA, rowa, cola);
    wbExport(input_file2.c_str(), hostB, rowb, colb);
    wbExport(output_file.c_str(), output, rowa, colb);

    
    return 0;


}
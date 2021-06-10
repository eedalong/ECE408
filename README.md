## Code Project For Course ECE408
This is code project for ECE408(Applied Parallel Programming). And you can use this project as a good start for your own lab implementation. Here's how you finish your own project.

### 1. Finish your own code based on template code
`mpx_template.cu` are template codes and you need to full fill these code files. For me, I add these implementations in `mpx_implement.cu`.

### 2. Compile & Run
    mkdir build
    cd build
    cmake ..
    make 

And executable files will be generated in under build/.

## Test Dataset
datasets for code test are under `test_data/`, you can use them to test your implementation with
     
     ./program -e <expected_output_file> -i <input_file_1>,<input_file_2> -o <output_file> -t <type>

The `<expected_output_file>` and `<input_file_n>` are the input and output files provided in the dataset. The `<output_file>` is the location you’d like to place the output from your program. The`<type>` is the output file type: `vector`, `matrix`, or `image`. If an MP does not expect an input or output, then pass none as the parameter.

For example, if you want test mp1_implementatiton, you can run
    
    #suppose you are in ./build directory

    ./MP1_Implement -e ../test_data/mp01/0/output.raw -i ../test_data/mp01/0/input0.raw,../test_data/mp01/0/input1.raw -o ../test_data/mp01/0/res.raw -t vector 





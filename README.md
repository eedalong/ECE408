Code For Course ECE408
## hello_world.cu
    nvcc hello_world.cu -o hello_world
    ./hello_world
*Notes:*
`cudaDeviceSynchronize` should be used to synchronize host with device, or we will not see any output from cuda kernel



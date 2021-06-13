## Parallel Scan


### Objective

Implement a kernel the performs parallel scan on a 1D list.
The scan operator used will be addition.
You should implement the work efficient kernel in Lecture 4.6.
Your kernel should be able to handle input lists of arbitrary length.
However, for simplicity, you can assume that the input list will be at most 2048 * 65,535 elements so that it can be handled by only one kernel launch.
The boundary condition can be handled by filling "identity value (0 for sum)" into the shared memory of the last block when the length is not a multiple of the thread block size.

### Prerequisites

Before starting this lab, make sure that:

* You have completed all week 4 lecture videos

### Instruction

Edit the code in the code tab to perform the following:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- copy results from device to host
- deallocate device memory
- implement the work efficient scan routine
- use shared memory to reduce the number of global memory accesses, handle the boundary conditions when loading input list elements into the shared memory

Instructions about where to place each part of the code is
demarcated by the `//@@` comment lines.



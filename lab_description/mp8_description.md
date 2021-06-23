
## Objective

The purpose of this lab is to get you familiar with using the CUDA streaming API by re-implementing a the vector addition machine problem to use CUDA streams.

## Prerequisites

Before starting this lab, make sure that:

* You have completed the vector addition machine problem

* You have completed all week 6 lecture videos

## Instruction

Edit the code in the code tab to perform the following:

* You MUST use at least 4 CUDA streams in your program, but 
  you may adjust it to be larger for larged datasets.

* Allocate device memory

* Interleave the host memory copy to device to hide 

* Initialize thread block and kernel grid dimensions

* Invoke CUDA kernel

* Copy results from device to host asynchronously

Instructions about where to place each part of the code is
demarcated by the `//@@` comment lines.


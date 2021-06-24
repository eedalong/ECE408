
## Objective

The purpose of this lab is to get you familiar with both the scatter and gather patterns.


## Instructions

In the provided source code you will find a function named `s2g_cpu_scatter`.
This function implements a simple scatter pattern on CPU.
It loops over an input array, then for each input element it performs some computation (`outInvariant(...)`), loops over the output array, does some more computation (`outDependent(...)`), and accumulates to the output element.

* Edit the function `s2g_cpu_gather` to implement a gather version of `s2g_cpu_scatter` on CPU. Compile and test the code.
* Edit the kernel `s2g_gpu_scatter_kernel` to implement a scatter version of `s2g_cpu_scatter` on the GPU, and edit the function `s2g_gpu_scatter` to launch the kernel you implemented.
* Edit the kernel `s2g_gpu_gather_kernel` to implement a gather version of `s2g_cpu_gather` on the GPU, and edit the function `s2g_gpu_gather` to launch the kernel you implemented.


Instructions about where to place each part of the code is
demarcated by the `//@@` comment lines.

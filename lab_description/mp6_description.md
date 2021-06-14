
## Objective

The lab's objective is to implement a tiled image convolution using both shared and constant memory as discussed in class.
Like what's discussed in class, we will have a constant 5x5 convolution mask, but will have arbitrarily sized image (We will assume the image dimensions are greater than 5x5 in this Lab). 

To use the constant memory for the convolution mask, you can first transfer the mask data  to the device. 
Assume you decided to name the pointer to the device array for the mask M.
As described in Lecture 3-5, you can use `const float * __restrict__ M` as one of the parameters during your kernel launch.
This informs the compiler that the contents of the mask array are constants and will only be accessed through pointer variable `M`.
This will enable the compiler to place the data into constant memory and allow the SM hardware to aggressively cache the mask data at runtime.

Convolution is used in many fields, such as image processing where it is used for
image filtering. A standard image convolution formula for a 5x5
convolution filter `M` with an Image `I` is:

![Equation](http://latex.codecogs.com/png.latex?%5Cfn_jvn%20P_%7Bi%2Cj%2Cc%7D%20%3D%20%5Csum_%7Bx%3D0%7D%5E%7B4%7D%20%5Csum_%7By%3D0%7D%5E%7B4%7D%20I_%7Bi&plus;x-2%2Cj&plus;y-2%2Cc%7D%20M_%7Bx%2Cy%7D)

where `P_{i,j,c}` is the output pixel at position `i,j` in channel `c`, `I_{i,j,c}` is the input pixel at `i,j` in channel `c`
(the number of channels will always be 3 for this MP corresponding to the RGB values), and `M_{x,y}` is
the mask at position `x,y`.

## Prerequisites

Before starting this lab, make sure that:

* You have completed all week 3 lecture videos

## Input Data

The input is an interleaved image of `height x width x channels`.
By interleaved, we mean that the the element `I[y][x]` contains three values representing the RGB channels.
This means that to index a particular element’s value, you will have to do something like:

        index = (yIndex*width + xIndex)*channels + channelIndex;

For this assignment, the channel index is 0 for R, 1 for G, and 2 for B. So, to access the G value of `I[y][x]`, you should use the linearized expression `I[(yIndex*width+xIndex)*channels + 1]`.

For simplicity, you can assume that `channels` is always set to `3`.


## Instruction

Edit the code in the code tab to perform the following:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- copy results from device to host
- deallocate device memory
- implement the tiled 2D convolution kernel with adjustments for channels
- use shared memory to  reduce the number of global accesses, handle the boundary conditions in when loading input list elements into the shared memory

Instructions about where to place each part of the code is
demarcated by the `//@@` comment lines.


## Pseudo Code

Your sequential pseudo code would look something like

        maskWidth := 5
        maskRadius := maskWidth/2 # this is integer division, so the result is 2
        for i from 0 to height do
          for j from 0 to width do
            for k from 0 to channels
              accum := 0
              for y from -maskRadius to maskRadius do
                for x from -maskRadius to maskRadius do
                  xOffset := j + x
                  yOffset := i + y
                  if xOffset >= 0 && xOffset < width &&
                     yOffset >= 0 && yOffset < height then
                    imagePixel := I[(yOffset * width + xOffset) * channels + k]
                    maskValue := K[(y+maskRadius)*maskWidth+x+maskRadius]
                    accum += imagePixel * maskValue
                  end
                end
              end
              # pixels are in the range of 0 to 1
              P[(i * width + j)*channels + k] = clamp(accum, 0, 1)
            end
          end
        end

where `clamp` is defined as

        def clamp(x, start, end)
          return min(max(x, start), end)
        end


## Input Format

For people who are developing on their own system.
The images are stored in PPM (`P6`) format, this means that you can (if you want) create your own input images.
The easiest way to create image is via external tools. You can use tools such as `bmptoppm`.
The masks are stored in a CSV format.
Since the input is small, it is best to edit it by hand.

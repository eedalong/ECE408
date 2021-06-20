import numpy as np
#import torch 


def readPPM(file_path):
    inputFile = open("../test_data/mp06/0/input0.ppm", encoding=None)
    firstLine = inputFile.readline()
    print(firstLine)
    secondLine = inputFile.readline()
    print(secondLine)
    shape = inputFile.readline()[:-1]
    # N, C, H, W 
    shape = shape.split()
    print(shape[0], shape[1], "")
    shape = [1, 3] + [int(item) for item in shape]
    print("shape is ", shape)
    total_size = 1
    for dim in shape:
        total_size *= dim
    print("total size is ", total_size)
    depth = inputFile.readline()
    print(depth)
    position = inputFile.tell()
    data = inputFile.read(total_size)
    image = np.zeros(shape)
    for row in range(shape[2]):
        for col in range(shape[3]):
            for channel in range(3):
                image[0][channel][row][col] = ord(data[(row * shape[3] + col) * 3 + channel]) / 255.0
    
    print(image[0])


readPPM("")
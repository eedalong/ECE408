import numpy as np
#import torch 

def getLine(data, offset):
    current_offset = offset
    while current_offset < len(data) and data[current_offset] != '\n':
        current_offset += 1
    
    return data[offset:current_offset], current_offset + 1
def readPPM(file_path):
    inputFile = open("../test_data/mp06/0/input0.ppm", 'rb')
    data = inputFile.read()
    firstLine, position = getLine(data, 0)
    print(firstLine)
    
    secondLine, position = getLine(data, position)
    print(secondLine)
    shape, position = getLine(data, position)
    print(shape)
    
    # N, C, H, W 
    shape = shape.split()
    print(shape[0], shape[1], "")
    shape = [1, 3] + [int(item) for item in shape]
    print("shape is ", shape)
    total_size = 1
    for dim in shape:
        total_size *= dim
    print("total size is ", total_size)
    depth, position = getLine(data, position)
    data = data[position: -1]
    image = np.zeros(shape)
    for row in range(shape[2]):
        for col in range(shape[3]):
            for channel in range(3):
                image[0][channel][row][col] = ord(data[(row * shape[3] + col) * 3 + channel]) / 255.0
    
    print(image[0])


readPPM("")
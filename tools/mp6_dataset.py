import numpy as np
import os 
import torch 

def getLine(data, offset):
    current_offset = offset
    while current_offset < len(data) and data[current_offset] != ord('\n'):
        current_offset += 1
    
    return data[offset:current_offset], current_offset + 1

def readPPM(file_path):
    inputFile = open(file_path, 'rb')
    data = inputFile.read()
    firstLine, position = getLine(data, 0)
    print(firstLine)
    
    secondLine, position = getLine(data, position)
    print(secondLine)
    shape, position = getLine(data, position)
    print(shape)
    
    # N, C, H, W 
    shape = shape.decode()
    shape = shape.split()
    shape.reverse()
    print(shape[0], shape[1], "")
    channel_num = 3
    shape = [1, channel_num] + [int(item) for item in shape]
    print("shape is ", shape)
    total_size = 1
    for dim in shape:
        total_size *= dim
    print("total size is ", total_size)
    depth, position = getLine(data, position)
    if data[-1] == ord('\n'):
        data = data[position: -1]
    else:
        data = data[position: ]
        
    print(f"check data[0:10]:\t{data[:10]}")
    image = np.zeros(shape)
    #assert len(data) == total_size, f"data length {len(data)} not equal to total size {total_size})"
    for row in range(shape[2]):
        for col in range(shape[3]):
            for channel in range(channel_num):
                image[0][channel][row][col] = int(data[(row * shape[3] + col) * 3 + channel]) / 255.0
    #print(f"check image {image[0][0][:5][:5]}")
    return torch.from_numpy(image)

def readInput(dir_path = "../build")->torch.Tensor:
    file_path = os.path.join(dir_path, "input0.ppm")
    return readPPM(file_path)
    

def readMask(dir_path = "../build"):
    file_path = os.path.join(dir_path, "input1.raw")
    inputFile = open(file_path, 'r')
    firstLine = inputFile.readline()
    shape = firstLine.split()
    shape = [1, 1] + [int(item) for item in shape]
    mask = np.zeros(shape)
    for index in range(shape[2]):
        data = [float(item) for item in inputFile.readline().split()]
        mask[0][0][index] = np.array(data)
    print("check mask \n", mask[0][0])
    return torch.from_numpy(mask)


def readExpectation(dir_path = "../build"):
    file_path = os.path.join(dir_path, "output.ppm")
    expect = readPPM(file_path)
    return expect

def readOutput(dir_path = "../build"):
    file_path = os.path.join(dir_path, "res.ppm")
    expect = readPPM(file_path)
    return expect


image = readInput()

sample = image[0][0]
mask = readMask()
output = readOutput()
expectation = readExpectation()
print(expectation[0][0][:5][:5])
print("++"* 30)
print(output[0][0][:5][:5])
print("++" * 30)
conv2d = torch.nn.Conv2d(1, 1, 5, stride=1, padding=2, bias=False, padding_mode='zeros')
print(f"conv2d shape {conv2d.weight.shape}")
conv2d.weight.data = mask
res = conv2d(image)
print("result and expectation macthes: ", torch.equal(output, expectation))



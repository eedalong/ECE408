import numpy as np
import os 
import torch 
'''
0.741176, 0.74902, 0.52549, 0.282353, 0.0823529,                                                                                                                    
0.376471, 0.933333, 0.729412, 0.721569, 0.94902,                                                                                                                    
0.886275, 0.701961, 0.121569, 0.101961, 0.45098,                                                                                                                    
0.47451, 0.807843, 0.25098, 0.12549, 0.317647,                                                                                                                      
0.588235, 0.6, 0.105882, 0.270588, 0.780392,                                                                                                                        
check mask                                                                                                                                                          
0.0114654, 0.0347546, 0.0354712, 0.00214977, 0.0906485,                                                                                                             
0.00143318, 0.0530276, 0.0146901, 0.0533859, 0.0118237,                                                                                                             
0.0748836, 0.027947, 0.0537442, 0.0641347, 0.00967395,                                                                                                              
0.0637764, 0.0447868, 0, 0.0114654, 0.0892153,                                                                                                                      
0.0730921, 0.0494446, 0.00501612, 0.0566105, 0.0673594,                                                                                                             
check outputImage                                                                                                                                                   
0.221104, 0.248094, 0.418581, 0.414895, 0.345892,                                                                                                                   
0.228149, 0.296501, 0.447234, 0.469453, 0.38529,                                                                                                                    
0.309048, 0.323117, 0.512451, 0.536325, 0.487769,                                                                                                                   
0.330894, 0.416483, 0.569247, 0.51843, 0.499735,                                                                                                                    
0.329266, 0.299235, 0.43253, 0.442911, 0.354963, 
'''
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
    print(shape[0], shape[1], "")
    channel_num = 1
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
    print(f"check image {image[0][0][:5][:5]}")
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
mask = readMask()
output = readOutput()
expectation = readExpectation()
conv2d = torch.nn.Conv2d(1, 1, 5, stride=1, padding=2, bias=False, padding_mode='zeros')
print(f"conv2d shape {conv2d.weight.shape}")
conv2d.weight.data = mask
res = conv2d(image)

print(res[0][0][0:5][0:5])
print("++" * 30)
print(output[0][0][0:5][0:5])
print("++" * 30)
print(expectation[0][0][0:5][0:5])



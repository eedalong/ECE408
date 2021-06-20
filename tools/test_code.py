import numpy as np
import os 
import torch 
'''
check inputImage                                                                                                                                                    
0.772549, 0.666667, 0.933333, 0.243137, 0.780392,                                                                                                                   
0.227451, 0.192157, 0.192157, 0.556863, 0.505882,                                                                                                                   
0.886275, 0.839216, 0.423529, 0.286275, 0.392157,                                                                                                                   
0.752941, 0.133333, 0.941176, 0.760784, 0.352941,                                                                                                                   
0.745098, 0.333333, 0.678431, 0.764706, 0.266667,                                                                                                                   
check mask                                                                                                                                                          
0.0528709, 0.0594088, 0.0144969, 0.0267197, 0.0295623,                                                                                                              
0.0355316, 0.0372371, 0.0690733, 0.0719159, 0.0611143,                                                                                                              
0.0494599, 0.0719159, 0.0582717, 0.0571347, 0.0687891,                                                                                                              
0.0133599, 0.00938033, 0.0252985, 0.0275725, 0.00852757,                                                                                                            
0.0227402, 0.0133599, 0.0426379, 0.0722001, 0.00142126,                                                                                                             
check outputImage                                                                                                                                                   
0.258984, 0.288038, 0.314156, 0.374602, 0.379852,                                                                                                                   
0.260108, 0.362079, 0.430922, 0.479349, 0.491968,                                                                                                                   
0.308136, 0.440931, 0.560244, 0.527713, 0.559303,                                                                                                                   
0.320995, 0.390854, 0.495568, 0.45771, 0.496076,                                                                                                                    
0.284003, 0.430685, 0.44912, 0.445971, 0.511511,
'''
def getLine(data, offset):
    current_offset = offset
    while current_offset < len(data) and data[current_offset] != ord('\n'):
        current_offset += 1
    
    return data[offset:current_offset], current_offset + 1

def readImage(dir_path = "../build")->torch.Tensor:
    file_path = os.path.join(dir_path, "input0.ppm")
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



image = readImage()
mask = readMask()
conv2d = torch.nn.Conv2d(1, 1, 5, stride=1, padding=2, bias=False, padding_mode='zeros')
print(f"conv2d shape {conv2d.weight.shape}")
conv2d.weight.data = mask

res = conv2d(image)
print(res[0][0][:5][:5])


import numpy as np
import os 
import torch 
'''
0.827451, 0.894118, 0.368627, 0.67451, 0.0784314,                                                                                                                   
0.929412, 0.835294, 0.513726, 0.027451, 0.65098,                                                                                                                    
0.32549, 0.407843, 0.0666667, 0.768627, 0.333333,                                                                                                                   
0.0666667, 0.909804, 0.835294, 0.505882, 0.360784,                                                                                                                  
0.545098, 0.576471, 0.686275, 0.596078, 0.278431,                                                                                                                   
check mask                                                                                                                                                          
0.0707965, 0.000856409, 0.0673708, 0.0573794, 0.0431059,                                                                                                            
0.0556666, 0.0382529, 0.0533828, 0.0679418, 0.046817,                                                                                                               
0.0316871, 0.041964, 0.0105624, 0.0465315, 0.0288324,                                                                                                               
0.0308307, 0.0191265, 0.0687982, 0.0256923, 0.0336854,                                                                                                              
0.0639452, 0.0245504, 0.0365401, 0.00513845, 0.0305452,                                                                                                             
check outputImage                                                                                                                                                   
0.179706, 0.253416, 0.377636, 0.34308, 0.293586,                                                                                                                    
0.216848, 0.325041, 0.429285, 0.323339, 0.359815,                                                                                                                   
0.251283, 0.363556, 0.530957, 0.470207, 0.498425,                                                                                                                   
0.20004, 0.41222, 0.503145, 0.481567, 0.520705,                                                                                                                     
0.162024, 0.260382, 0.387172, 0.479142, 0.571821, 
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
    shape = [1, 3] + [int(item) for item in shape]
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
    assert len(data) == total_size, f"data length {len(data)} not equal to total size {total_size})"
    for row in range(shape[2]):
        for col in range(shape[3]):
            for channel in range(3):
                image[0][channel][row][col] = int(data[(row * shape[3] + col) * 3 + channel]) / 255.0
    
    return torch.from_numpy(image)

def readMask(dir_path = "../build"):
    file_path = os.path.join(dir_path, "input1.raw")
    inputFile = open(file_path, 'r')
    firstLine = inputFile.readline()
    shape = firstLine.split()
    shape = [1] + [int(item) for item in shape]
    mask = np.zeros(shape)
    for index in range(shape[1]):
        data = [float(item) for item in inputFile.readline().split()]
        mask[0][index] = np.array(data)
    return torch.from_numpy(mask)



image = readImage()
mask = readMask()
conv2d = torch.nn.Conv2d(1, 1, 5, stride=1, padding=2, bias=False, padding_mode='zeros')
conv2d.weight.data = mask
res = conv2d(image[:][0][:][:])
print(res)


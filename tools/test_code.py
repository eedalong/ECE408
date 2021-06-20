import numpy as np
import os 
import torch 
'''
check input image channel 0                                                                                                                                         
0.235294, 0.905882, 0.341176, 0.0392157, 0.360784,                                                                                                                  
0.113725, 0.172549, 0.117647, 0.0705882, 0.811765,                                                                                                                  
0.431373, 0.105882, 0.254902, 0.564706, 0.796079,                                                                                                                   
0.364706, 0.87451, 0.513726, 0.160784, 0.411765,                                                                                                                    
0.882353, 0.141176, 0.756863, 0.0980392, 0.254902,                                                                                                                  
check mask                                                                                                                                                          
0.024234, 0.077415, 0.025581, 0.026254, 0.081117,                                                                                                                   
0.044093, 0.02962, 0.033659, 0.070347, 0.004039,                                                                                                                    
0.0138, 0.073713, 0.015483, 0.032986, 0.072366,                                                                                                                     
0.035005, 0.049142, 0.0138, 0.075059, 0.082464,                                                                                                                     
0.027264, 0.024907, 0.054864, 0.010771, 0.00202,                                                                                                                    
check output                                                                                                                                                        
0.107759, 0.0885229, 0.225104, 0.23904, 0.273509,                                                                                                                   
0.154377, 0.232944, 0.324418, 0.372219, 0.373346,                                                                                                                   
0.267032, 0.273857, 0.410648, 0.455386, 0.488581,                                                                                                                   
0.21982, 0.259866, 0.372017, 0.425497, 0.451548,                                                                                                                    
0.252441, 0.372227, 0.409951, 0.453717, 0.463637,                                                                                                                   
check solution image                                                                                                                                                
0.107759, 0.0885229, 0.225104, 0.23904, 0.273509,                                                                                                                   
0.154377, 0.232944, 0.324418, 0.372219, 0.373346,                                                                                                                   
0.267032, 0.273857, 0.410648, 0.455386, 0.488581,                                                                                                                   
0.21982, 0.259866, 0.372017, 0.425497, 0.451548,                                                                                                                    
0.252441, 0.372227, 0.409951, 0.453717, 0.463637,                                                                                                                   
check expected image                                                                                                                                                
0.109804, 0.0901961, 0.227451, 0.239216, 0.27451,                                                                                                                   
0.156863, 0.235294, 0.32549, 0.372549, 0.376471,                                                                                                                    
0.270588, 0.27451, 0.411765, 0.458824, 0.490196,                                                                                                                    
0.223529, 0.262745, 0.372549, 0.427451, 0.454902,                                                                                                                   
0.254902, 0.372549, 0.411765, 0.454902, 0.466667,      
'''
"""
tensor([[0.1098, 0.0902, 0.2275, 0.2392, 0.2745, 0.3294, 0.3569, 0.3843, 0.2706,                                                                                    
         0.2510, 0.2392, 0.1804, 0.1765, 0.2588, 0.3490, 0.3137, 0.3765, 0.3373,                                                                                    
         0.3255, 0.4078, 0.3373, 0.3765, 0.3765, 0.3176, 0.2941, 0.2863, 0.3098,                                                                                    
         0.2980, 0.3216, 0.3059, 0.2863, 0.2902, 0.2510, 0.2784, 0.1843, 0.2627,                                                                                    
         0.3020, 0.2549, 0.2353, 0.2941, 0.3569, 0.2784, 0.3843, 0.4157, 0.3804,                                                                                    
         0.4471, 0.4078, 0.3882, 0.3765, 0.3725, 0.3059, 0.2863, 0.3020, 0.2980,                                                                                    
         0.2275, 0.2275, 0.2863, 0.2824, 0.2549, 0.2627, 0.3255, 0.2510, 0.1961,                                                                                    
         0.1804],                                                                                                                                                   
        [0.1569, 0.2353, 0.3255, 0.3725, 0.3765, 0.4941, 0.4824, 0.4549, 0.4549,                                                                                    
         0.3059, 0.2314, 0.2588, 0.2706, 0.3569, 0.3922, 0.4196, 0.4235, 0.4235,                                                                                    
         0.4549, 0.4471, 0.4745, 0.3765, 0.3922, 0.4078, 0.3804, 0.4784, 0.4118,                                                                                    
         0.4078, 0.3882, 0.3961, 0.3725, 0.3412, 0.3569, 0.2706, 0.3059, 0.3412,                                                                                    
         0.3373, 0.3059, 0.3961, 0.4196, 0.4000, 0.3725, 0.4039, 0.4863, 0.4980,                                                                                    
         0.4980, 0.4627, 0.4627, 0.4667, 0.4392, 0.3961, 0.3804, 0.4235, 0.3098,                                                                                    
         0.3216, 0.2863, 0.2667, 0.3529, 0.2902, 0.3529, 0.4392, 0.4000, 0.2902,                                                                                    
         0.2275],                                                
"""
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
                image[0][channel][row][col] = byte(data[(row * shape[3] + col) * 3 + channel]) / 255.0
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



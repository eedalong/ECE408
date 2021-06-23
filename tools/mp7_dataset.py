import numpy as np

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
    
    # C, H, W 
    shape = shape.decode()
    shape = shape.split()
    shape.reverse()
    print(shape[0], shape[1], "")
    channel_num = 3
    shape = [channel_num] + [int(item) for item in shape]
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
    for row in range(shape[1]):
        for col in range(shape[2]):
            for channel in range(channel_num):
                image[channel][row][col] = int(data[(row * shape[2] + col) * 3 + channel]) / 255.0
    #print(f"check image {image[0][0][:5][:5]}")
    return image


inputImage = readPPM("../build/input.ppm")
inputImage = 255.0 * inputImage
inputImage = inputImage.astype(np.ubyte)
gray_image = inputImage[0] * 0.21 + inputImage[1] * 0.71 + inputImage[2] * 0.07
gray_image = gray_image.astype(np.ubyte)
print(gray_image)

globalHist = [0] * 256
globalCDF = [0] * 256
for row in range(gray_image.shape[0]):
    for col in range(gray_image.shape[1]):
        globalHist[gray_image[row][col]] += 1

print(globalHist)



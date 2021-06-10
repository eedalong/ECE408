import numpy as np
inputA = open("test_data/mp02/7/input0.raw")
inputB = open("test_data/mp02/7/input1.raw")
output = open("test_data/mp02/7/output.raw")

inputA_array = []
inputB_array = []
output_array = []
next(inputA)
next(inputB)
next(output)
for line in inputA:
    line = line[:-1].split()
    data = [float(item) for item in line]
    inputA_array.append(data)

for line in inputB:
    line = line[:-1].split()
    data = [float(item) for item in line]
    inputB_array.append(data)
    
for line in output:
    line = line[:-1].split()
    data = [float(item) for item in line]
    output_array.append(data)

inputA_array  = np.array(inputA_array, dtype = np.float)
inputB_array = np.array(inputB_array, dtype = np.float)
output_array = np.array(output_array, dtype = np.float)

res = np.dot(inputA_array, inputB_array)
print(inputA_array.shape, inputB_array.shape)
res_0_1 = 0
for index in range(inputA_array[0].shape[0]):
    res_0_1 += inputA_array[0][index] * inputB_array[index][1]

print(res_0_1)

print(res)
import torch
import numpy as np

x = torch.rand(5, 3)
#print(x)

data = [[1,2], [3,4]]
x_data = torch.tensor(data)
np_arr = np.array(data)
x_np = torch.from_numpy(np_arr)
#print(x_data)
#print(data)
#print(f"tensor : \n{x_np}")

# describe tensor's dimension by tuple
rank = (3, 4)
rand_tensor = torch.rand(rank)
one_tensor = torch.ones(rank)
zero_tensor = torch.zeros(rank)

#print(rand_tensor)
#print(one_tensor)
#print(zero_tensor)
#print(rand_tensor + one_tensor)

# attribute of tensor

tensor = torch.rand(5, 3)
print(tensor)
#print(f"Shape of tensor: {tensor.shape}")
#print(f"Data type of tensor: {tensor.dtype}")
#print(f"Device tensor is stored on: {tensor.device}") # cpu

if torch.cuda.is_available():
    tensor = tensor.to("cuda")
#print(f"Device tensor is stored on: {tensor.device}") # gpu

# slicing
#print(f"First row: {tensor[0]}")
#rint(f"First column: {tensor[:, 0]}")
#print(f"Last row: {tensor[:, -1]}")

# stacking

t1 = torch.cat([tensor, tensor, tensor], dim=1)
#print(t1)


# matrix multiplication
# 'tensor.T' returns transpose of tensor

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
#print(f"tensor : {y1}")
#print(f"tensor : {y2}")
#print(f"tensor : {y3}")

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#print(f"tensor : {z1}")
#print(f"tensor : {z2}")
#print(f"tensor : {z3}")

agg = tensor.sum()
agg_item = agg.item()
#print(agg)
#print(agg_item)

# tensor to NumPy arr

t = torch.ones(5)
#print(f"f: {t}")
n = t.numpy()
#print(f"n: {n}")

# changes in tensor reflects for numPy arr
t.add_(1)
#print(f"f: {t}")
#print(f"n: {n}")

# numPy arr to tensor

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")



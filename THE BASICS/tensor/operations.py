"""
텐서의 연산
"""
import torch

tensor = torch.rand(3, 4)

# There are over 100 tensor operations.
# Each of operations can be run on GPU(higher speeds).
# So, We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# check out tensor.device
print(f"tensor is stored on: {tensor.device} \n")

# indexing and slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First col: {tensor[:, 0]}")  # : means select all element
print(f"Last col: {tensor[..., -1]} \n")  # ... = :
tensor[:, 1] = 0
print(tensor)

# joining
t1 = torch.cat([tensor, tensor, tensor], dim=1)  # dim is adhesive surface
print(t1, "\n")  # 2 dimension is same

t2 = torch.cat([tensor, tensor, tensor], dim=0)
print(t2, "\n")

# Arithmetic operations
# 1. computes the matrix multiplication between two tensors.
y1 = tensor @ tensor.T  # T means transpose
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)  # make tensor shape like y1
torch.matmul(tensor, tensor.T, out=y3)  # out means save variable

print(f"y1: {y1}")
print(f"y2: {y2}")
print(f"y3: {y3} \n")

# 2. computes the element-wise product.
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print(f"z1: {z1}")
print(f"z2: {z2}")
print(f"z3: {z3} \n")

# 3. sum
agg = tensor.sum()
agg_item = agg.item()  # convert to a Python value using item()
print(agg_item, type(agg_item))  # python float

# 4. In-place operation(add_, copy_, t_(),...)
print(f"{tensor} \n")
tensor.add_(5)  # no other variable
print(tensor)

"""
텐서를 초기화 하는 방법
"""

# first, import torch to use method torch.tensor()
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)  # convert list to tensor

print(x_data)

# Tensors can be initialized randomly or by one, zero
shape = (2, 3,)  # shape is a tuple of tensor dimensions.
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")  # f can print variable between {}
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

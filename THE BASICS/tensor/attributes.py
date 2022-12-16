"""
텐서의 속성
"""

import torch

tensor = torch.rand(3, 4)  # row=3, col=4

print(f"Shape of tensor: {tensor.shape}")  # size
print(f"Datatype of tensor: {tensor.dtype}")  # torch.float32
print(f"Device tensor is stored on: {tensor.device}")  # cpu or cuda

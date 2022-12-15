"""
텐서를 초기화 하는 방법
"""

# first, import torch to use method torch.tensor()
import torch

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)  # convert list to tensor

print(x_data)

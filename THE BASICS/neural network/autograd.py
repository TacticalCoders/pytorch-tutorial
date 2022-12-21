"""
torch.autograd를 통한 자동 미분
"""

# most frequently used algorithm is back propagation.
# parameters are adjusted according to the gradient of loss function
# torch.autograd supports automatic computation of gradient

import torch

# simplest one-layer neural net
x = torch.ones(5)
y = torch.zeros(3)
# w and b are parameters, which need to optimize.
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# backpropagation function is stored in grad_fn
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# To optimize weight, need to computue the derivatives of our loss function withe respect to parameters.
# To compute derivatives, call loss.backward()
loss.backward()
# we can only obtain the grad for the leaf nodes of computational graph.
print(w.grad)
print(b.grad)

# By default, all tensors with requires_grad=True are tracking their computational history.
# However, there are some cases when we do not need to do that(only want to do forward)
# we can stop tracking, by with torch.no_grad():
z = torch.matmul(x, w) + b
print(z.requires_grad)  # True

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)  # False

# also, use  detach() method on the tensor.
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z.requires_grad)  # False

import torch
import numpy as np

a = np.random.RandomState(0)
a = a.randn(3, 4)

x = torch.tensor(a, dtype=torch.float32)
print(x)

# 使用x的0轴的元素来填充y，tensor（）里的值是x的0轴(行)元素对应的索引
# torch.tensor（）里的两组索引对应x的0轴的前两个元素，里面的值是y的行索引
# 由于torch.tensor()的值是做索引，所以它的形状要和x保持一致。
y = torch.zeros(4, 4).scatter_(0, torch.tensor([[2, 0, 3, 1], [1, 0, 2, 0]]), x)
print(y)

z = torch.zeros(4, 10).scatter_(1, torch.tensor([[5], [8], [2], [3]]), 2)
print(z)

tensor = torch.tensor([5, 3, 8, 6])
print(tensor.view(-1, 1))
print(tensor.size(0))
print(torch.zeros(tensor.size(0), 10))

tensor_out = torch.zeros(tensor.size(0), 10).scatter_(1, tensor.view(-1, 1), 1)
print(tensor_out)
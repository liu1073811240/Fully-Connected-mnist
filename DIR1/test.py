import torch

a1 = torch.randn(10, 2)
a2 = torch.randn(1, 10, 2)

b = torch.nn.BatchNorm1d(2)
c = torch.nn.InstanceNorm1d(2)  # 要求维度是三维

print(b(a1))
print(c(a2))
print(c(a2)[0])
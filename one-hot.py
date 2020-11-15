import numpy as np
import torch


def one_hot(w, h, arr):
    z = np.zeros([w, h])  # 四行七列
    # print(z)

    for i in range(w):  # 4
        j = int(arr[i])  # 拿到数组里面的数字
        # print(j)
        z[i][j] = 1
    return z


if __name__ == '__main__':
    arr = np.array([2, 5, 6, 5])

    a = one_hot(len(arr), max(arr)+1, arr)
    # print(a)
    # print(a.argmax(1))

    label_onehot = torch.zeros(len(arr), max(arr) + 1)
    print(label_onehot.numpy())
    label_onehot[torch.arange(len(arr)), arr] = 1
    print(label_onehot.numpy().argmax(1))

    tensor = torch.tensor(arr, dtype=torch.long)
    tensor_out = torch.zeros(tensor.size(0), max(arr) + 1).scatter_(1, tensor.view(-1, 1), 1)
    print(tensor_out.numpy())
    print(tensor_out.numpy().argmax(1))



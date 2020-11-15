import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from dataset_sampling import MyDataset
from torch.optim import sgd, adam, adagrad, rmsprop, adadelta, adamax, adamw, sparse_adam, asgd


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(784, 512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = torch.reshape(x, [x.size(0), -1])
        y1 = self.layer1(x)

        y2 = self.layer2(y1)

        y3 = self.layer3(y2)

        self.y4 = self.layer4(y3)

        out = torch.softmax(self.y4, 1)

        return out


if __name__ == '__main__':
    batch_size = 100
    # 加载数据集
    data_path = r"D:\PycharmProjects\2020-08-25-全连接神经网络\MNIST_IMG"
    save_params = "../mnist_params.pth"
    save_net = "./mnist_net.pth"

    train_data = MyDataset(data_path, True)
    test_data = MyDataset(data_path, False)

    train_loader = data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Net().to(device)
    # net.load_state_dict(torch.load(save_params))
    # net = torch.load(save_net).to(device)

    loss_fn = nn.MSELoss()

    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.5)
    optimizer = torch.optim.Adam(net.parameters(),)

    # weight_decay:表示L2正则化系数
    # optimizer = adagrad.Adagrad(net.parameters())
    # optimizer = adadelta.Adadelta(net.parameters())
    # optimizer = rmsprop.RMSprop(net.parameters())
    # optimizer = sgd.SGD(net.parameters(), 1e-3)
    # optimizer = adam.Adam(net.parameters())

    a = []
    b = []
    plt.ion()
    net.train()
    for epoch in range(2):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            output = net(x)
            # print(output)
            # print(output.shape)
            # print(y)

            y = torch.zeros(y.cpu().size(0), 10).scatter_(1, y.cpu().reshape(-1, 1), 1).to(device)
            # print(y)
            # print(y.size(0))

            # 加正则化
            # L1 = 0
            # L2 = 0
            # for params in net.parameters():
            #     L1 += torch.sum(torch.abs(params))
            #     L2 += torch.sum(torch.pow(params, 2))
            #
            # loss = loss_fn(output, y)
            # loss1 = loss + 0.001*L1
            # loss2 = loss + 0.001*L2
            # loss = 0.2*loss1 + 0.8*loss2

            loss = loss_fn(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                a.append(i + epoch*(len(train_data) / batch_size))
                b.append(loss.item())
                plt.clf()
                plt.plot(a, b)
                plt.pause(1)
                print("epoch:{}, batch:{}/600, loss:{:.3f}".format(epoch, int(i), loss.item()))

        torch.save(net.state_dict(), "./mnist_params.pth")
        torch.save(net, "./mnist_net.pth")

    net.eval()
    eval_loss = 0
    eval_acc = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        out = net(x)
        y = torch.zeros(y.cpu().size(0), 10).scatter_(1, y.cpu.reshape(-1, 1), 1)

        loss = loss_fn(out, y)
        print("Test_loss:{:.3f}".format(loss.item()))

        print(y.size(0))
        print(loss.item())
        print("==")

        eval_loss += loss.item()*y.size(0)

        arg_max = torch.argmax(out, 1)
        y = y.argmax(1)

        eval_acc += (arg_max == y).sum().item()

    mean_loss = eval_loss / len(test_data)
    mean_acc = eval_acc / len(test_data)

    print("loss:{:.3f}, Acc:{:.3f}".format(mean_loss, mean_acc))

















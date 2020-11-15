import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils import data
import matplotlib.pyplot as plt
from dataset_sampling import MyDataset
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter("./logs")  # 生成日志文档
# 调出可视化窗口命令：tensorboard --logdir=路径

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc4 = nn.Linear(128, 10)

    def forward(self, x, epoch):
        x = x.reshape(x.size(0), -1)
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)

        # writer.add_histogram("out", y, epoch)  # 收集输出、轮次
        # writer.add_histogram("weight", self.fc4.weight, epoch)  # 收集权重，轮次
        return y


if __name__ == '__main__':
    batch_size = 100
    # 加载本地数据集
    data_path = r"D:\PycharmProjects\2020-08-25-全连接神经网络\MNIST_IMG"
    save_params = "./mnist_params.pth"
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
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=0)

    net.train()
    for epoch in range(2):
        train_loss = 0
        train_acc = 0
        for i, (x, y) in enumerate(train_loader):
            # y=torch.zeros(y.size(0),10).scatter_(1,y.reshape(-1,1),1)
            x = x.to(device)
            y = y.to(device)

            out = net(x, epoch)

            loss = loss_fn(out, y)
            out = torch.argmax(out, 1)

            train_loss += loss.item()  # 一张图片的损失

            train_acc += torch.sum(torch.eq(y.cpu(), out.detach().cpu()))

            optim.zero_grad()
            loss.backward()
            optim.step()

        train_avgloss = train_loss / len(train_loader)  # 一轮次的损失除以一轮的训练数据长度
        train_avgacc = train_acc / len(train_loader)

        test_loss = 0
        test_acc = 0
        for i, (x, y) in enumerate(test_loader):
            # y = torch.zeros(y.size(0),10).scatter_(1,y.reshape(-1,1),1)
            x = x.to(device)
            y = y.to(device)

            out = net(x, epoch)
            loss = loss_fn(out, y)

            out = torch.argmax(out, 1)
            test_loss += loss.item()
            test_acc += torch.sum(torch.eq(y.cpu(), out.detach().cpu()))

        test_avgloss = test_loss / len(test_loader)
        test_avgacc = test_acc / len(test_loader)

        print("epoch:{},train_loss:{:.3f},test_loss:{:.3f}".format(epoch, train_avgloss, test_avgloss))
        print("epoch:{},train_acc:{:.3f}%,test_acc:{:.3f}%".format(epoch, train_avgacc, test_avgacc))

        # writer.add_scalars("loss", {"train_loss": train_avgloss, "test_loss": test_avgloss}, epoch)
        # writer.add_scalars("acc", {"train_acc": train_avgacc, "test_acc": test_avgacc}, epoch)
        torch.save(net.state_dict(), save_params)
        # torch.save(net, save_net)

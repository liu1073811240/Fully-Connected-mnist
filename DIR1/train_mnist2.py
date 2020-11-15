import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils import data
import matplotlib.pyplot as plt
from dataset_sampling import MyDataset
from tensorboardX import SummaryWriter

writer = SummaryWriter("./log")
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

        # writer.add_histogram("out", y, epoch)
        # writer.add_histogram("weight", self.fc4.weight, epoch)
        return y


if __name__ == '__main__':
    batch_size = 100
    data_path = r"D:\PycharmProjects\2020-08-25-全连接神经网络\MNIST_IMG"
    save_params = "./mnist_params.pth"
    save_net = "./mnist_net.pth"

    train_data = MyDataset(data_path, True)
    test_data = MyDataset(data_path, False)

    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Net().to(device)
    # net.load_state_dict(torch.load(save_params))
    # net = torch.load(save_net).to(device)

    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters())

    net.train()
    for epoch in range(2):
        train_loss = 0
        train_acc = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            out = net(x, epoch)
            loss = loss_fn(out, y)

            out1 = torch.argmax(out, 1)

            train_loss += loss.item()

            train_acc += torch.sum(torch.eq(y.cpu(), out1.detach().cpu()))

            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 10 == 0:
                print("epoch:{}, batch:{}, loss:{:.3f}".format(epoch, int(i), loss.item()))

        train_avgloss = train_loss / len(train_loader)
        train_avgacc = train_acc / len(train_loader)

        test_loss = 0
        test_acc = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            out = net(x, epoch)
            loss = loss_fn(out, y)

            out2 = torch.argmax(out, 1)
            test_loss += loss.item()
            test_acc += torch.sum(torch.eq(y.cpu, out.detach().cpu()))

        test_avgloss = test_loss / len(test_loader)
        test_avgacc = test_acc / len(test_loader)

        print("epoch:{}, train_loss:{:.3f}, test_loss:{:.3f}".format(epoch, train_avgloss, test_avgloss))
        print("epoch:{}, train_acc:{:.3f}%, test_acc:{:.3f}%".format(epoch, train_avgacc, test_avgacc))

        torch.save(net.state_dict(), save_params)









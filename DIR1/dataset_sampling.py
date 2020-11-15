from  torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch


class MyDataset(Dataset):
    def __init__(self, main_dir, is_train=True):
        self.dataset = []
        data_filename = "TRAIN" if is_train else "TEST"
        for i, cls_filename in enumerate(os.listdir(os.path.join(main_dir, data_filename))):
            # print(i)
            # print(os.listdir(os.path.join(main_dir)))
            # print(os.listdir(os.path.join(main_dir, data_filename)))
            # 循环获得每个类别文件夹下的数字图片
            for img_data in os.listdir(os.path.join(main_dir, data_filename, cls_filename)):
                self.dataset.append([os.path.join(main_dir, data_filename, cls_filename, img_data), i])
                # 装图片路径可以节省内存，避免列表装了所有图片导致内存爆炸
                # print(self.dataset)

    def __len__(self):
        return len(self.dataset)  # 获取数据集长度个数，方便迭代

    def __getitem__(self, index):
        data = self.dataset[index]  # 根据索引来取[[图片数据路径，标签]...]
        image_data = self.image_preprocess(Image.open(data[0]))  # 拿到图片数据路径打开并得到图片数据
        label_data = data[1]  # 拿到图片标签
        return image_data, label_data

    def image_preprocess(self, x):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])(x)


if __name__ == '__main__':
    data_path = r"D:\PycharmProjects\2020-08-25-全连接神经网络\MNIST_IMG"
    dataset = MyDataset(data_path, True)

    dataloader = DataLoader(dataset, 128, shuffle=True, )
    for x, y in dataloader:
        print(x.shape)
        print(y.shape)




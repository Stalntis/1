

import os
os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'

import os
import sys
import json
import time
import torch
import torch.optim as optim
from torchvision import transforms, datasets
import torch.nn as nn
from main import Net1,Net2,Net3,FusionNet
from utils import train_and_val,plot_acc,plot_loss
import numpy as np




if __name__ == '__main__':


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device ))

    if not os.path.exists('./weight'):
        os.makedirs('./weight')

    BATCH_SIZE = 32

    # 定义数据增强和预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


    # 加载数据集
    train_dataset1 = datasets.ImageFolder('data/train1', transform=data_transform["train"])
    train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=32, shuffle=True)

    train_dataset2 = datasets.ImageFolder('data/train2', transform=data_transform["train"])
    train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2, batch_size=32, shuffle=True)

    train_dataset3 = datasets.ImageFolder('data/train3', transform=data_transform["train"])
    train_loader3 = torch.utils.data.DataLoader(dataset=train_dataset3, batch_size=32, shuffle=True)

    len_train = len(train_dataset1)


    val_dataset1 = datasets.ImageFolder('data/test1', transform=data_transform["val"])
    val_loader1 = torch.utils.data.DataLoader(val_dataset1, batch_size=BATCH_SIZE, shuffle=False)

    val_dataset2 = datasets.ImageFolder('data/test2', transform=data_transform["val"])
    val_loader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=BATCH_SIZE, shuffle=False)

    val_dataset3 = datasets.ImageFolder('data/test3', transform=data_transform["val"])
    val_loader3 = torch.utils.data.DataLoader(val_dataset3, batch_size=BATCH_SIZE, shuffle=False)


    len_val = len(val_dataset1)



    # 定义网络和优化器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net1 = Net1()
    net2 = Net2()
    net3 = Net3()
    fusion_net = FusionNet()
    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    optimizer = optim.SGD(fusion_net.parameters(), lr=0.003, momentum=0.9, weight_decay=5e-4)

    # optimizer = optim.Adam(net.parameters(), lr=0.003)  # 设置优化器和学习率
    epoch = 50

    history = train_and_val(epoch, net1,net2,net3,fusion_net, train_loader1, train_loader2,train_loader3, len_train,val_loader1, val_loader2, val_loader3,len_val,
                            loss_function, optimizer, device)


    plot_loss(np.arange(0,epoch), history)
    plot_acc(np.arange(0,epoch), history)



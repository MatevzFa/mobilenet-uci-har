import os
import sys
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import (AvgPool2d, BatchNorm2d, Conv2d, Linear, MaxPool2d,
                      Module, ReLU, Sequential, Softmax)

import loading

ActivT = Optional[Callable[[], Module]]


def make_conv_pool_activ(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    activation: ActivT = None,
    pool_size: Optional[int] = None,
    pool_stride: Optional[int] = None,
    **conv_kwargs
):
    layers = [Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)]
    if activation:
        layers.append(activation())
    if pool_size is not None:
        layers.append(MaxPool2d(pool_size, stride=pool_stride))
    return layers


class Classifier(Module):
    def __init__(
        self, convs: Sequential, linears: Sequential, use_softmax: bool = True
    ):
        super().__init__()
        self.convs = convs
        self.linears = linears
        self.softmax = Softmax(1) if use_softmax else Sequential()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.convs(inputs)
        return self.softmax(self.linears(outputs.view(outputs.shape[0], -1)))


class BasicBlock(Module):
    def __init__(self, ins, outs, shortcut=False):
        super().__init__()
        stride = 2 if shortcut else 1
        self.mainline = Sequential(
            *make_conv_pool_activ(ins, outs, 3, ReLU,
                                  padding=1, stride=stride),
            *make_conv_pool_activ(outs, outs, 3, padding=1)
        )
        self.relu1 = ReLU()
        self.shortcut = (
            Sequential(*make_conv_pool_activ(ins, outs, 1, stride=stride))
            if shortcut
            else Sequential()
        )

    def forward(self, input_):
        return self.relu1(self.mainline(input_) + self.shortcut(input_))


class ResNet18(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(3, 16, 3, ReLU, padding=1),
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 16),
            BasicBlock(16, 32, True),
            BasicBlock(32, 32),
            BasicBlock(32, 32),
            BasicBlock(32, 64, True),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            AvgPool2d(8)
        )
        linears = Sequential(Linear(64, 10))
        super().__init__(convs, linears)


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.mainline = Sequential(
            *make_conv_pool_activ(in_planes, planes, 1, stride=stride),
            BatchNorm2d(planes, eps=0.001),
            ReLU(),
            *make_conv_pool_activ(planes, planes, 3, padding=1),
            BatchNorm2d(planes, eps=0.001),
            ReLU(),
            *make_conv_pool_activ(planes, self.expansion * planes, 1),
            BatchNorm2d(self.expansion * planes, eps=0.001)
        )
        self.relu1 = ReLU()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                *make_conv_pool_activ(
                    in_planes, self.expansion * planes, 1, stride=stride
                ),
                BatchNorm2d(self.expansion * planes, eps=0.001)
            )
        else:
            self.shortcut = Sequential()

    def forward(self, input_):
        return self.relu1(self.mainline(input_) + self.shortcut(input_))


class ResNet50(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(
                3, 64, 7, ReLU, pool_size=3, pool_stride=2, padding=3, stride=2
            ),
            BatchNorm2d(64, eps=0.001),
            Bottleneck(64, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 128, stride=2),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 256, stride=2),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 256),
            Bottleneck(1024, 512, stride=2),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
            AvgPool2d(7)
        )
        linears = Sequential(Linear(2048, 1000))
        super().__init__(convs, linears)


def _make_seq(in_channels, out_channels, c_kernel_size, gc_stride, gc_kernel_size=3):
    return Sequential(
        *make_conv_pool_activ(
            in_channels,
            out_channels,
            c_kernel_size,
            bias=False,
            padding=(c_kernel_size - 1) // 2,
        ),
        BatchNorm2d(out_channels, eps=0.001),
        ReLU(),
        Conv2d(
            out_channels,
            out_channels,
            gc_kernel_size,
            bias=False,
            stride=gc_stride,
            padding=(gc_kernel_size - 1) // 2,
            groups=out_channels,
        ),
        BatchNorm2d(out_channels, eps=0.001),
        ReLU()
    )


class MobileNet(Classifier):
    def __init__(self):
        convs = Sequential(
            _make_seq(3, 32, 3, 1),
            _make_seq(32, 64, 1, 2),
            _make_seq(64, 128, 1, 1),
            _make_seq(128, 128, 1, 2),
            _make_seq(128, 256, 1, 1),
            _make_seq(256, 256, 1, 2),
            _make_seq(256, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 2),
            _make_seq(512, 1024, 1, 1),
            *make_conv_pool_activ(1024, 1024, 1, padding=0, bias=False),
            BatchNorm2d(1024, eps=0.001),
            ReLU(),
            AvgPool2d(2)
        )
        linears = Sequential(Linear(1024, 10))
        super().__init__(convs, linears)


def model(file=None):
    net = MobileNet().float()
    if file is not None:
        net.load_state_dict(torch.load(file))
    return net


def train(lr, nepochs, batch_size):
    path = Path(os.getenv("HAR_PIPELINE_PATH")) / \
        "Batch/Data/Original-Data/UCI-HAR-Dataset/Processed-Data"

    y_train_txt = path / ".." / "y_train.txt"

    nn_X_train = torch.tensor(loading.compose("train", np.float32))
    nn_y_train = torch.tensor(np.loadtxt(y_train_txt, dtype=int))

    train_data = torch.utils.data.TensorDataset(nn_X_train, nn_y_train)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    net = model()

    print(net)

    print(f"train shape = {nn_X_train.shape}")

    criterion = nn.CrossEntropyLoss().float()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    every = 3
    running_loss = 0.0
    for epoch in range(nepochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % every == (every-1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / every))
                running_loss = 0.0

    print("Done training.")

    torch.save(net.state_dict(), file)

    return net


def eval(net, batch_size):
    path = Path(os.getenv("HAR_PIPELINE_PATH")) / \
        "Batch/Data/Original-Data/UCI-HAR-Dataset/Processed-Data"

    y_test_txt = path / ".." / "y_test.txt"
    nn_X_test = torch.tensor(loading.compose("test", np.float32))
    nn_y_test = torch.tensor(np.loadtxt(y_test_txt, dtype=int))

    test_data = torch.utils.data.TensorDataset(nn_X_test, nn_y_test)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    predictions = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            predictions.extend(list(predicted))
            all_labels.extend(list(labels))

    print('Accuracy of the network on test data: %d %%' % (
        100 * correct / total))

    conf_mat = np.zeros((6, 6), dtype=int)
    for p in range(len(predictions)):
        conf_mat[all_labels[p]-1][predictions[p]-1] += 1

    conf_mat = np.divide(conf_mat, np.sum(conf_mat, axis=1))

    label_names = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]
    df_cm = pd.DataFrame(conf_mat, label_names, label_names)

    plt.figure(figsize=(5, 5))
    sn.set(font_scale=.5)
    snplt = sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    snplt.figure.savefig(f"{file}.pdf", bbox_inches='tight')
    snplt.figure.savefig(f"{file}.png", bbox_inches='tight')


if __name__ == '__main__':
    assert len(sys.argv) > 2
    do_train = sys.argv[1] == "train"
    file = sys.argv[2]

    if do_train:
        net = train(lr=0.4, nepochs=5, batch_size=128)
    else:
        net = model(file)

    eval(net, batch_size=128)

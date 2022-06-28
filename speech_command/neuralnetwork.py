from turtle import forward
from torch import strided
import torch.nn as nn

class Sound(nn.Module):

    def __init__(self, kernel_size=100, stride=50, out_channels=30):
        super(Sound, self).__init__()

        self.conv =  nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        self.ReLU = nn.ReLU()
        self.MaxPool = nn.MaxPool1d(kernel_size=10)
        self.Linear = nn.Linear(30*31, 35)
        self.log = nn.LogSoftmax(dim=1)
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv(x)
        out = self.ReLU(out)
        out = self.MaxPool(out)
        out = out.view(-1, 31*30)
        out = self.Linear(out)
        out = self.log(out)
        return out

class SoundDeep(nn.Module):

    def __init__(self, kernel_size=100, stride=25, out_channels=30):
        super(SoundDeep, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
        self.Filter = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv_2 =  nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels*2,
            kernel_size=3,
        )
        self.ReLU = nn.ReLU()
        self.MaxPool = nn.MaxPool1d(kernel_size=4)
        self.batch1 = nn.BatchNorm1d(out_channels)
        self.batch2 = nn.BatchNorm1d(2*out_channels)
        self.Linear = nn.Linear(60*9, 35)
        self.log = nn.LogSoftmax(dim=1)
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv(x)
        out = self.ReLU(out)
        out = self.batch1(out)
        out = self.MaxPool(out)

        out = self.Filter(out)
        out = self.ReLU(out)
        out = self.MaxPool(out)

        out = self.conv_2(out)
        out = self.ReLU(out)
        out = self.batch2(out)
        out = self.MaxPool(out)

        out = out.view(-1, 60*9)
        out = self.Linear(out)
        out = self.log(out)
        return out
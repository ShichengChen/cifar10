import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
'''
class Residual_v2_bottleneck(nn.Module):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual_v2_bottleneck, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(channels=channels // 4, kernel_size=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels=channels // 4, kernel_size=3, strides=strides, padding=1, use_bias=False)
        self.bn3 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(channels=channels, kernel_size=1, use_bias=False)
        self.bn4 = nn.BatchNorm()  # add this

        if not same_shape:
            self.conv4 = nn.Conv2D(channels=channels, kernel_size=1, strides=strides, use_bias=False)

    def forward(self, F, x):
        out = self.conv1(self.bn1(x))  # remove relu
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = self.bn4(out)  # add this

        if not self.same_shape:
            x = self.conv4(x)
        return out + x


class ResNet164_v2(nn.Module):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet164_v2, self).__init__()
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # block 1
            net.add(nn.Conv2D(channels=64, kernel_size=3, padding=1, strides=1, use_bias=False))
            # block 2
            for _ in range(27):
                net.add(Residual_v2_bottleneck(channels=64))
            # block 3
            net.add(Residual_v2_bottleneck(128, same_shape=False))
            for _ in range(26):
                net.add(Residual_v2_bottleneck(channels=128))
            # block4
            net.add(Residual_v2_bottleneck(256, same_shape=False))
            for _ in range(26):
                net.add(Residual_v2_bottleneck(channels=256))
            # block 5
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print("Block %d output %s" % (i + 1, out.shape))
        return out
'''

"""
ResNet 18
"""
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Residual(nn.Module):
    def __init__(self, inc,outc, same_shape=True, **kwargs):
        super(Residual, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(inc,outc, kernel_size=3, padding=1, stride=strides)  # w,h or w/2,h/2
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc,outc, kernel_size=3, padding=1)  # not change w,h
        self.bn2 = nn.BatchNorm2d(outc)
        if not same_shape:
            self.conv3 = nn.Conv2d(inc,outc, kernel_size=1, stride=strides)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)


class ResNet(nn.Module):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__()
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.Sequential()
            # block 1
            net.add(nn.Conv2d(3,32,3,padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU())
            # block 2
            for _ in range(3):
                net.add(Residual(32,32))
            # block 3
            net.add(Residual(32,64, same_shape=False))
            for _ in range(2):
                net.add(Residual(64,64))
            # block 4
            net.add(Residual(64,128, same_shape=False))
            for _ in range(2):
                net.add(Residual(128,128))
            # block 5
            net.add(nn.AvgPool2d(kernel_size=8))
            net.add(Reshape(128,))
            self.last = nn.Linear(128,num_classes)

    def forward(self, x):
        out = x
        out = self.net(out)
        out = out.view(out.shape[0],out.shape[1])
        return self.last(out)
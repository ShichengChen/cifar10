from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import os
import argparse

from model.resnet import *


resumefile = 'model/pyramid'  # name of checkpoint
continueTrain = False  # whether use checkpoint
sampleCnt=0
USEBOARD = False
quan=False
if(USEBOARD):writer = SummaryWriter()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use specific GPU


from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
if(USEBOARD):writer = SummaryWriter(log_dir='../conditioned-wavenet/runs/'+str(current_time)+'deltacifar10resnet',comment="uwavenet")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


net = ResNet(10)
net = net.to(device)

sds_ = [torch.zeros(param.shape,device='cuda') for param in net.parameters() if param.requires_grad]
cnt=0
for param in net.parameters():
    if param.requires_grad:
        dist = torch.distributions.normal.Normal(loc=param, scale=sds_[cnt])
        param.data=(dist.sample().view(param.shape)).clone()
        cnt += 1

if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)

iteration = 0
start_epoch=0
if continueTrain:  # if continueTrain, the program will find the checkpoints
    if os.path.isfile(resumefile):
        print("=> loading checkpoint '{}'".format(resumefile))
        checkpoint = torch.load(resumefile)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resumefile, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resumefile))


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        #############delta rule
        cnt = 0
        #print(sds_[cnt][0])
        for param in net.parameters():
            if param.requires_grad:
                dist = torch.distributions.normal.Normal(loc=param.data, scale=sds_[cnt])
                #dist = torch.distributions.normal.Normal(loc=torch.zeros(sds_[cnt].shape).to('cuda'), scale=sds_[cnt])
                param.data=(dist.sample().view(param.shape)).clone()
                cnt += 1
        #############

        loss.backward()

        ###########################################
        cnt = 0
        for param in net.parameters():
            if param.requires_grad:
                sds_[cnt]=(0.1*(sds_[cnt]+torch.abs(param.grad.data)*0.1)).clone()
                cnt += 1
        ##############################################33



        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        global iteration
        iteration += 1

    print('train Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if (USEBOARD):
        writer.add_scalar('train resnet acc', 100.*correct/total, iteration)
        writer.add_scalar('train resnet loss', float(loss),iteration)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if (USEBOARD):
        writer.add_scalar('test resnet acc', 100. * correct / total, epoch)
        writer.add_scalar('test resnet loss', test_loss/(batch_idx+1), epoch)

    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state = {'epoch': epoch,
                 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'iteration': iteration}
        torch.save(state, resumefile)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    scheduler.step()
    train(epoch)
    test(epoch)
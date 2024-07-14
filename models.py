from collections import OrderedDict, defaultdict
import torch
from torch import nn  
from torch.nn import functional as F
from typing import List
from torch.nn.functional import relu, avg_pool2d

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int=1, out_feature=False) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.out_feature = out_feature

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                            stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_ = []
        out = relu(self.bn1(self.conv1(x)))
        if self.out_feature==True: output_.append(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        if self.out_feature==True: output_.append(out)
        if self.out_feature==False: return out
        else: return out, output_


class ResNet(nn.Module):
    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        
        self.layer4_1 = self._make_layer(block, nf * 8, num_blocks[3]/2, stride=2, out_feature=True)
        self.layer4_2 = self._make_layer(block, nf * 8, num_blocks[3]/2, stride=1, out_feature=True)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       nn.ReLU(),
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4_1,
                                       self.layer4_2
                                       )
        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int, out_feature=False) -> nn.Module:
        if num_blocks>1:
            strides = [stride] + [1] * (num_blocks - 1)
        else: strides = [stride]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, out_feature))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        
        out = relu(self.bn1(self.conv1(x))) # 64, 32, 32
        if hasattr(self, 'maxpool'):
            out = self.maxpool(out)
        out_feature = {}
        out = self.layer1(out)  # -> 64, 32, 32
        out = self.layer2(out)  # -> 128, 16, 16
        out = self.layer3(out)  # -> 256, 8, 8
        out, output_ = self.layer4_1(out)  # -> 512, 4, 4
        cnt = 0
        for i in output_:
            cnt+=1
            out_feature[f'layer4.0.conv{cnt}'] = i
        out, output_ = self.layer4_2(out)
        cnt = 0
        for i in output_:
            cnt+=1
            out_feature[f'layer4.1.conv{cnt}'] = i
        out = avg_pool2d(out, int(out.shape[2])) # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512
        out = self.classifier(feature)
        out_feature['linear'] = out
        return out, out_feature

def resnet18(dataset: str, nf: int=64) -> ResNet:
    nclasses = 0
    if dataset == 'seq-cifar10' or dataset == 'rot-mnist':
        nclasses = 10
    elif dataset == 'seq-cifar100':
        nclasses = 100
    elif dataset == 'seq-tinyimg':
        nclasses = 200
    elif dataset == 'ImageNet':
        nclasses = 1000 
    print(nclasses)
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)

def resnet34(dataset: str, nf: int=64) -> ResNet:
    if dataset == 'seq-cifar10' or dataset == 'rot-mnist':
        nclasses = 10
    elif dataset == 'seq-cifar100':
        nclasses = 100
    elif dataset == 'seq-tinyimg':
        nclasses = 200
    model = ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf)
    return model

def resnet50(dataset: str, nf: int=64) -> ResNet:
    if dataset == 'seq-cifar10' or dataset == 'rot-mnist':
        nclasses = 10
    elif dataset == 'seq-cifar100':
        nclasses = 100
    elif dataset == 'seq-tinyimg':
        nclasses = 200
    model = ResNet(Bottleneck, [3, 4, 6, 3], nclasses, nf)
    return model

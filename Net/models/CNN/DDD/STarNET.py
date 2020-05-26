import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolu tion with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class SarNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(SarNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
    
class sarnet2D(nn.Module):
    """Constructs SarNet model.
    
    Args: 
        x (tensor): model input
        num_classes (int,optional): number of identification classes. default: 2
        layer_sizes (list, optional): describes depth of network in terms of number
            of convolutional layers. default: [2,2,2,2]
        block_type (class, optional): what block types to use to build the conv. default: BasicBlock
    """
    
    def __init__(self, block_type=BasicBlock, num_classes=2, layer_sizes=[2,2,2,2]):
        super(sarnet2D, self).__init__()

        self.res2D = SarNet(block_type,layer_sizes)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.res2D(x)
        x = self.linear(x) 

        return x

class sarnet3D(nn.Module):
    """Constructs SarNet model.
    
    Args: l
        x (tensor): model input
        fcount(int): number of frames in the model input
        num_classes (int,optional): number of identification classes. default: 2
        layer_sizes (list, optional): describes depth of network in terms of number
            of convolutional layers. default: [2,2,2,2]
        block_type (class, optional): what block types to use to build the conv. default: BasicBlock
    """
    
    def __init__(self, block_type=BasicBlock, num_classes=2, layer_sizes=[2,2,2,2]):
        super(sarnet3D, self).__init__()
        
        self.res2D = SarNet(block_type, layer_sizes)
        self.linear1 = nn.Linear(512, 512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512,100)
        self.linear3 = nn.Linear(100,num_classes)

    def forward(self, x):
        out = []
        for k in range(len(x)):
            out.append(self.res2D(x[k]))
        model_out = self.linear3(self.relu(self.linear2(self.relu(self.linear1(out))))) 

        return model_out


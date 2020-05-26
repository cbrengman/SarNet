import math
import torch.nn as nn
from torch.nn.modules.utils import _triple


class ST_Conv(nn.Module):
    """
    Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output. Modified from "A Closer Look at Spatiotemporal Convolutions
    for Action Recognition, Tran et. al. 2018"

    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of filters produced by the convolution
        kernel_size (int or tuple): Size of the convolutional filter
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during 
                                          their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ST_Conv, self).__init__()

        #Convert to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        
        #Set parameters up in iterable of form [t,d,d]
        #Where slot [0] is time domain and slots [1] and [2] are spatial domain
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        #Compute the number of intermediary channels (M) using formula 
        #from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        #Setup Basic Block
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x
    
class ST_Basic_Block(nn.Module):
    """
    Single block based on ResNet. 
        
    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output produced by the block.
        kernel_size (int or tuple): Size of the convolutional filters.
        downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
    """
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(ST_Basic_Block, self).__init__()

        self.downsample = downsample
        padding = kernel_size//2

        if self.downsample:
            #Reduce dimensionality by 2x by setting stride=2
            self.downsampleconv = ST_Conv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
            self.conv1 = ST_Conv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = ST_Conv(in_channels, out_channels, kernel_size, padding=padding)

        self.conv2 = ST_Conv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))    
        out = self.bn2(self.conv2(res))

        if self.downsample:
            residual = self.downsamplebn(self.downsampleconv(x))
            
        out += residual
        out = self.relu(out)

        return out
    
class STsarnet(nn.Module):
    """ 
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: ST_Basic_Block. 
    """
    def __init__(self, layer_sizes, block_type=ST_Basic_Block):
        super(STsarnet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = ST_Conv(1, 64, [1, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = ST_Basic_Block(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling 
        # inside the first block
        self.conv3 = ST_Basic_Block(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = ST_Basic_Block(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = ST_Basic_Block(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)
        x - x.view(x.size[0], -1)
        
        return x

class STsarnet_Classifier(nn.Module):
    """
    Constructs STsarnet model
        
    Args:
        num_classes(int): Number of classes in the data (default: 2)
        layer_sizes (tuple): An iterable containing the number of blocks in each layer (default: [2,2,2,2])
        block_type (Module, optional): Type of block that is to be used to form the layers. Default: ST_Basic_Block. 
    """
    def __init__(self, num_classes=2, layer_sizes=[2,2,2,2], block_type=ST_Basic_Block):
        super(STsarnet_Classifier, self).__init__()

        self.res2plus1d = STsarnet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.res2plus1d(x)
        x = self.linear(x) 

        return x   
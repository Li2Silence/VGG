## VERY DEEP CONVOLUTIONAL NETWORKS（VGG）

虽然AlexNet证明深层神经网络卓有成效，但它没有提供一个通用的模板来指导后续的研究人员设计新的网络。AlexNet网络包含5个卷积层，3个全连接层，只是定义了层，而在V`GG中，首次提出通过使用循环和子程序，将神经网络的卷积、全连接等层堆叠成块进行使用。

### VGG改进

#### 小卷积核

在ALexNet中，作者使用了$11 \times11$，$5 \times5$的卷积核来提取图像的特征，但大卷积核计算量大，并不能详细提取出图像的语义信息。在VGG网络中，作者使用了成组出现的小卷积核，如下图所示，使用了两个$3 \times 3$的卷积核代替了一个$5 \times 5$的卷积，来提取图像的语义信息。

![1672897686321](1672897686321.jpg)

#### 参数量

使用一组小卷积核来替代大卷积核可以明显减少参数量，避免模型出现过拟合的现象。

两个$3\times3$的卷积核与一个$5\times5$卷积核的参数量对比，如下式所示，可以看出小卷积核组的参数明显比大卷积核参数小！
$$
3 \times3:        3\times3\times3\times96\times2=5184\\
5 \times5:        5\times5\times3\times96=7220
$$

#### 结果

VGG网络在ILSVRC-2012 数据集上取的了下图的效果，表格中第一列代表VGG的模型规模。

<img src="1672899299116.jpg" alt="1672899299116" style="zoom:80%;" />

下图为VGG网络与其他网络的对比，可以看出使用VGG网络可以超过其他方法。

<img src="image-20230105141908044.png" alt="image-20230105141908044" style="zoom:80%;" />

### 模型结构

#### VGG块

经典卷积神经网络的基本组成部分是下面的这个序列：

1. 带填充以保持分辨率的卷积层；
1. 非线性激活函数，如ReLU；
1. 池化层，如最大池化层。

而一个VGG块与之类似，由一系列卷积层组成，后面再加上用于空间下采样的最大池化层。在最初的VGG论文中，作者使用了带有$3 × 3$卷积核、填充为1（保持高度和宽度）的卷积层，和带有$2 × 2$池化窗口、步幅为2（每个块后的分辨率减半）的最大池化层。这样做的意义是，一方面可以减少参数，另一方面相当于进行了更多的非线性映射，可以增加网络的拟合/表达能力。

```python
import torch
from torch import nn
def vgg_block(num_convs, in_channels, out_channels): # 卷积层的数量num_convs、输入通道的数量in_channels和输出通道的数量out_channels
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

如下图所示，最左边为AlexNet网络，包含5个卷积和3个全连接。中间则为VGG块，每个块包含1个卷积层、1个池化层和1个激活函数层。最右边就是一个VGG网络，包含多个VGG块，

<img src="架构.jpg" alt="架构" style="zoom:67%;" />

#### 网络结构

在所有的VGG网络中，VGG16和VGG19最为常见，在此文档中，如果没有特殊说明，均以VGG16网络进行讲解。VGG16包含13个卷积层，13个卷积层分成了5个VGG块，3个全连接层。

<img src="VGG网络.jpg" alt="VGG网络" style="zoom:80%;" />

#### 流程

<img src="shape.jpg" alt="shape" style="zoom: 70%;" />

假设输入的图像尺寸为$X=[16,3,224,224]$,其中16为batch size，3为通道数，224为图像的宽和高。

第一个卷积块：图像经过两个卷积核为$3\times3$，步长为1，padding=1的卷积层，图像的尺寸变成了$X=[16,64,224,224]$

在第一个卷积块结束后，经过一个最大化池化层，滤波器为2x2，步长为2，图像的尺寸变成了$X=[16,64,112,112]$

第二个卷积块：图像经过两个卷积核为$3\times3$，步长为1，padding=1的卷积层，图像的尺寸变成了$X=[16,128,112,112]$

在第二个卷积块结束后，经过一个最大化池化层，滤波器为2x2，步长为2，图像的尺寸变成了$X=[16,128,56,56]$

第三个卷积块：图像经过三个卷积核为$3\times3$，步长为1，padding=1的卷积层，图像的尺寸变成了$X=[16,256,56,56]$

在第三个卷积块结束后，经过一个最大化池化层，滤波器为2x2，步长为2，图像的尺寸变成了$X=[16,256,28,28]$

第四个卷积块：图像经过三个卷积核为$3\times3$，步长为1，padding=1的卷积层，图像的尺寸变成了$X=[16,512,28,28]$

在第四个卷积块结束后，经过一个最大化池化层，滤波器为2x2，步长为2，图像的尺寸变成了$X=[16,512,14,14]$

第五个卷积块：图像经过三个卷积核为$3\times3$，步长为1，padding=1的卷积层，图像的尺寸变成了$X=[16,512,14,14]$

在第五个卷积块结束后，经过一个最大化池化层，滤波器为2x2，步长为2，图像的尺寸变成了$X=[16,512,7,7]$

然后Flatten()，将数据拉平成向量，变成一维$512\times7\times7=25088$

再经过两层1x1x4096，一层1x1x1000的全连接层（共三层），经ReLU激活

最后通过softmax输出1000个预测结果

### 参考文献

Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.

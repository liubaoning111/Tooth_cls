import os
import numpy as np
 
from PIL import Image
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import transforms
 
 
def main():
    #这个下面放置你网络的代码，因为载入权重的时候需要读取网络代码，这里我建议直接从自己的训练代码中原封不动的复制过来即可，我这里因为跑代码使用的是Resnet，所以这里将resent的网络复制到这里即可
    class BasicBlock(nn.Module):
        expansion = 1
 
        def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                   kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                   kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channel)
            self.downsample = downsample
 
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
 
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
 
            out = self.conv2(out)
            out = self.bn2(out)
 
            out += identity
            out = self.relu(out)
 
            return out
 
    class Bottleneck(nn.Module):
        """
        注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
        但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
        这么做的好处是能够在top1上提升大概0.5%的准确率。
        可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
        """
        expansion = 4
 
        def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                     groups=1, width_per_group=64):
            super(Bottleneck, self).__init__()
 
            width = int(out_channel * (width_per_group / 64.)) * groups
 
            self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                                   kernel_size=1, stride=1, bias=False)  # squeeze channels
            self.bn1 = nn.BatchNorm2d(width)
            # -----------------------------------------
            self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                   kernel_size=3, stride=stride, bias=False, padding=1)
            self.bn2 = nn.BatchNorm2d(width)
            # -----------------------------------------
            self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                                   kernel_size=1, stride=1, bias=False)  # unsqueeze channels
            self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
 
        def forward(self, x):
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
 
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
 
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
 
            out = self.conv3(out)
            out = self.bn3(out)
 
            out += identity
            out = self.relu(out)
 
            return out
 
    class ResNet(nn.Module):
 
        def __init__(self,
                     block,
                     blocks_num,
                     num_classes=5,
                     include_top=True,
                     groups=1,
                     width_per_group=64):
            super(ResNet, self).__init__()
            self.include_top = include_top
            self.in_channel = 64
 
            self.groups = groups
            self.width_per_group = width_per_group
 
            self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                                   padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.in_channel)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, blocks_num[0])
            self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
            self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
            self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
            if self.include_top:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
                self.fc = nn.Linear(512 * block.expansion, num_classes)
 
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
 
        def _make_layer(self, block, channel, block_num, stride=1):
            downsample = None
            if stride != 1 or self.in_channel != channel * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(channel * block.expansion))
 
            layers = []
            layers.append(block(self.in_channel,
                                channel,
                                downsample=downsample,
                                stride=stride,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
            self.in_channel = channel * block.expansion
 
            for _ in range(1, block_num):
                layers.append(block(self.in_channel,
                                    channel,
                                    groups=self.groups,
                                    width_per_group=self.width_per_group))
 
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
 
            if self.include_top:
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
 
            return x
 
    def resnet34(num_classes=2, include_top=True):
        # https://download.pytorch.org/models/resnet34-333f7ec4.pth
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
 
    def resnet50(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet50-19c8e357.pth
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
 
    def resnet101(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
 
    def resnext50_32x4d(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
        groups = 32
        width_per_group = 4
        return ResNet(Bottleneck, [3, 4, 6, 3],
                      num_classes=num_classes,
                      include_top=include_top,
                      groups=groups,
                      width_per_group=width_per_group)
 
    def resnext101_32x8d(num_classes=1000, include_top=True):
        # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
        groups = 32
        width_per_group = 8
        return ResNet(Bottleneck, [3, 4, 23, 3],
                      num_classes=num_classes,
                      include_top=include_top,
                      groups=groups,
                      width_per_group=width_per_group)
 
    net = resnet34(num_classes=2)
 
 
 
 
    device = torch.device("cpu")
    net.load_state_dict(torch.load("/home/liubn/0-deep-learning-for-image-processing-master/pytorch_classification/Test5_resnet/resNet34(banakeyaceweiruanzuzhi).pth", map_location=device))  # 载入训练的resnet模型权重，你将训练的模型权重放到当前文件夹下即可
 
    target_layers = [net.layer4[-1]] #这里是 看你是想看那一层的输出，我这里是打印的resnet最后一层的输出，你也可以根据需要修改成自己的
    print(target_layers)
    data_transform = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 导入图片
    img_path = "/home/liubn/0-deep-learning-for-image-processing-master/pytorch_classification/data_set/flower_data/val/4444/103.jpg"#这里是导入你需要测试图片
    image_size = 224#训练图像的尺寸，在你训练图像的时候图像尺寸是多少这里就填多少
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')#将图片转成RGB格式的
    img = np.array(img, dtype=np.uint8) #转成np格式
    img = center_crop_img(img, image_size) #将测试图像裁剪成跟训练图片尺寸相同大小的
 
    # [C, H, W]
    img_tensor = data_transform(img)#简单预处理将图片转化为张量
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0) #增加一个batch维度
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)
 
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.savefig('/home/liubn/0-deep-learning-for-image-processing-master/pytorch_classification/0-results/1-3.png')#将热力图的结果保存到本地当前文件夹
    plt.show()
 
 
if __name__ == '__main__':
    main()

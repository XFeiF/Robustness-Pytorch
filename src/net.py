import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision.models import resnet
from torchvision.models import vgg


class Vgg16(vgg.VGG):
    def __init__(self,  num_classes=10):
        super().__init__(vgg.make_layers(vgg.cfgs['D'], batch_norm=True), init_weights=False)
        self.load_state_dict(model_zoo.load_url(vgg.model_urls['vgg16_bn']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Res18(resnet.ResNet):
    def __init__(self, num_classes=10):
        super().__init__(resnet.BasicBlock, [2, 2, 2, 2])
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()


class NoiseRes18(Res18):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, train=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = self.maxpool(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.avgpool(f4)
        x = f5.view(f5.size(0), -1)
        x = self.fc(x)
        if train:
            return x, (f0, f1, f2, f3, f4, f5)
        else:
            return x


class DR18(nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_layer0 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.inv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.inv_conv1 = nn.ConvTranspose2d(64, 3, 3, 2, 1, 1)
        self.inv_bn1 = nn.BatchNorm2d(3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.inv_layer0(x) + x
        x = F.relu(x, True)
        x = self.inv_layer1(x) + self.upsample(x)
        x = F.relu(x, True)
        x = self.inv_conv1(x)
        x = self.inv_bn1(x)
        return torch.tanh(x)


class AE18(Res18):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = DR18()

    def forward(self, x, train=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        dc = self.decoder(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if train:
            return x, dc
        else:
            return x


class Mix18(AE18):
    def forward(self, x, train=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        dc = self.decoder(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        f = x.view(x.size(0), -1)
        x = self.fc(f)
        if train:
            return x, f, dc
        else:
            return x


'''
class DecodeRes18(nn.Module):
    def __init__(self):
        super().__init__()
        self.inv_layer0 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, 1),
            nn.BatchNorm2d(64)
        )
        self.inv_conv1 = nn.ConvTranspose2d(64, 3, 3, 2, 1, 1)
        self.inv_bn1 = nn.BatchNorm2d(3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.inv_layer0(x)+self.upsample(x)
        x = F.relu(x, True)
        x = self.inv_conv1(x)
        x = self.inv_bn1(x)
        return torch.tanh(x)


class AERes18(Res18):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder = DecodeRes18()

    def forward(self, x, train=False, deep=5):
        if deep != 5:
            raise ValueError('Temporary control')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1[0](x)
        dex = x
        x = self.layer1[1](x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if train:
            return x, self.decoder(dex)
        else:
            return x


class MixRes18(AERes18):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, train=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1[0](x)
        dex = x
        x = self.layer1[1](x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        f5 = self.avgpool(x)
        x = f5.view(f5.size(0), -1)
        x = self.fc(x)
        if train:
            return x, f5, self.decoder(dex)
        else:
            return x


class ResSparse(nn.Module):

    def __init__(self, block=resnet.BasicBlock, layers=[2, 2, 2, 2], sparse=512, num_classes=10):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, sparse, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(sparse, num_classes)

        temp = model_zoo.load_url(resnet.model_urls['resnet18'])
        pre_dict = {}
        for k, v in temp.items():
            if k[5] == '4':
                break
            pre_dict[k] = v
        mod_dict = self.state_dict()
        mod_dict.update(pre_dict)
        self.load_state_dict(mod_dict)

        for m in self.layer4.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

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

    def forward(self, x, train=False):
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
        feats = x
        x = self.fc(x)
        if train:
            return x, feats
        else:
            return x


class ResSparseN(ResSparse):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, train=False):
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
        x = x / x.norm(dim=1, keepdim=True)
        feats = x
        x = self.fc(x)
        if train:
            return x, feats
        else:
            return x


class MS(ResSparse):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, train=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        f5 = self.avgpool(f4)
        f5 = f5.view(f5.size(0), -1)
        x = self.fc(f5)
        if train:
            return x, f1, f2, f3, f4, f5
        else:
            return x
'''


class GauConv(nn.Module):
    def __init__(self, in_planes, kernel_size, sigma, padding=1, stride=1,
                 bias=False):
        super(GauConv, self).__init__()
        self.n = in_planes
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias
        # self._parameters = OrderedDict()
        self.sigmas = torch.tensor([sigma for _ in range(self.n)])

        s, c = self.kernel_size, self.kernel_size >> 1
        self.k = torch.tensor([[
            (i - c) ** 2 + (j - c) ** 2
            for i in range(s)]
            for j in range(s)]).float()

        self.base_ks = torch.empty(
            (self.n, 1, self.kernel_size, self.kernel_size))
        self.__gen__()

    def forward(self, x):
        if x.is_cuda:
            self.base_ks = self.base_ks.cuda()
        return F.conv2d(x, self.base_ks, padding=self.padding,
                        stride=self.stride, groups=self.n)

    def __gen__(self):
        sigmas_2 = torch.pow(self.sigmas, 2)
        pi = np.pi
        sigmas_2 = sigmas_2.reshape(self.n, 1, 1, 1)
        pre = 1 / (2 * pi * sigmas_2)
        self.base_ks[:] = self.k
        self.base_ks = self.base_ks / (-(2 * sigmas_2))
        self.base_ks = pre * torch.exp(self.base_ks)


class GResNet(resnet.ResNet):
    def __init__(self, num_classes=1000, block=resnet.Bottleneck):
        layers = [3, 4, 6, 3]
        super().__init__(block, layers)
        if block == resnet.BasicBlock:
            self.load_state_dict(
                model_zoo.load_url(resnet.model_urls['resnet18']))
        elif block == resnet.Bottleneck:
            if layers[1] == 4:
                self.load_state_dict(
                    model_zoo.load_url(resnet.model_urls['resnet50']))
            elif layers[1] == 8:
                self.load_state_dict(
                    model_zoo.load_url(resnet.model_urls['resnet152']))

        # new
        self.kernel_size = 7
        self.sigma = 1.5
        self.g_padding = 3

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()

        self.gaussian1 = self._make_gaussian_conv(64 * block.expansion,
                                                  self.kernel_size,
                                                  self.sigma,
                                                  self.g_padding)
        self.gaussian2 = self._make_gaussian_conv(128 * block.expansion,
                                                  self.kernel_size,
                                                  self.sigma,
                                                  self.g_padding)
        self.gaussian3 = self._make_gaussian_conv(256 * block.expansion,
                                                  self.kernel_size,
                                                  self.sigma,
                                                  self.g_padding)
        self.gaussian4 = self._make_gaussian_conv(512 * block.expansion,
                                                  self.kernel_size,
                                                  self.sigma,
                                                  self.g_padding)

    def _make_gaussian_conv(self, planes, kernel_size, sigma, padding=1, stride=1, bias=False):
        gaussian_conv = GauConv(planes, kernel_size, sigma, padding, stride, bias)
        return gaussian_conv

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.gaussian1(x)
        x = self.layer2(x)
        x = self.gaussian2(x)
        x = self.layer3(x)
        x = self.gaussian3(x)
        x = self.layer4(x)
        x = self.gaussian4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Res50(resnet.ResNet):
    def __init__(self, num_classes=10):
        super().__init__(resnet.Bottleneck, [3, 4, 6, 3])
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()


def get_net(args):
    if args['mid'] == 'res18':
        return Res18(num_classes=args['num_classes'])
    elif args['mid'] == 'nres18':
        return NoiseRes18(num_classes=args['num_classes'])
    elif args['mid'] == 'ares18':
        return AE18(num_classes=args['num_classes'])
    elif args['mid'] == 'mix18':
        return Mix18(num_classes=args['num_classes'])
    elif args['mid'] == 'vgg16':
        return Vgg16(num_classes=args['num_classes'])
    elif args['mid'] == 'g50':
        return GResNet(num_classes=args['num_classes'])
    elif args['mid'] == 'res50':
        return Res50(num_classes=args['num_classes'])
    else:
        raise ValueError('No net: {}'.format(args['mid']))


if __name__ == '__main__':
    pass

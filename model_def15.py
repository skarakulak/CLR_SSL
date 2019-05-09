import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# reference: https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_clust=2000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes) 
        self.cl_centers = nn.ParameterList([
            nn.parameter.Parameter(torch.Tensor(num_clust, 128)),
            nn.parameter.Parameter(torch.Tensor(num_clust, 256)),
            nn.parameter.Parameter(torch.Tensor(num_clust, 512))
        ])
        for l in self.cl_centers: nn.init.normal_(l)


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

    def get_cdist(self, x, c_ind):
        b,f,h,w = x.size()
        if a.ndimension()>2:
            # sample one pixel per image
            x = x.view(b,f,-1).transpose(1,2)
            rand_idx = torch.randint(0,x.size(1),(b,))
            r=rand_idx[:,None,None].expand(b,1,f)
            x = torch.gather(input=x, dim=1, index=r).squeeze(1)
        # k-means
        x_k = x.unsqueeze(1).expand(b,self.cl_centers[c_ind].size(0),f)
        c_k = self.cl_centers[c_ind].unsqueeze(0).expand(b,self.cl_centers[c_ind].size(0),f)
        c_dist = torch.min(((x_k-c_k)**2).sum(2), dim=1)[0].mean()
        return c_dist


    def forward(self, x, add_eps = None, return_c_dist=False, return_latent=False):
        c_dist = 0
        z_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        if return_c_dist: c_dist += self.get_cdist(x,0)
        if return_latent: z_list.append(x.detach())

        x = self.layer3(x)
        if return_c_dist: c_dist += self.get_cdist(x,1)
        if return_latent: z_list.append(x.detach())

        x = self.layer4(x)
        x = self.avgpool(x).view(x.size(0), 512)
        if return_c_dist: c_dist += self.get_cdist(x,2)
        if return_latent: z_list.append(x.detach())

        x = self.fc(x)

        if return_latent and return_c_dist:  return x, z_list, c_dist
        elif return_c_dist: return x, c_dist
        elif return_latent: return x, z_list
        else: return x


def resnet18(pretrained=False, num_clust=2000, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(
        BasicBlock, 
        [2, 2, 2, 2],
        num_clust=num_clust,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(pretrained=False, num_clust=2000, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(
        BasicBlock, 
        [3, 4, 6, 3], 
        num_clust=num_clust, 
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


# reference: https://github.com/pytorch/examples/blob/master/dcgan/main.py
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(512, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 32 x 32
            nn.ConvTranspose2d(128, 64,  4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 64 x 64
            nn.ConvTranspose2d(64,  3,   4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 128 x 128
        )

    def forward(self, input):
        output = self.main(input)
        return output[:,:,22:106,22:106]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 84 x 84
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 42 x 42
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 21 x 21
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 10 x 10
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 5 x 5
            nn.Conv2d(512, 512, 5, 1, 0, bias=False)#,
            #nn.Sigmoid()
            # state size. 512 x 1 x 1
        )
        self.latent_space_func = nn.Sequential(
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,256)
        )
        self.final_disc = nn.Sequential(
            nn.Linear(768,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, input, cl_centers):
        output = self.main(input).view(-1,512)
        output = torch.cat(
            (
                output,
                self.latent_space_func(cl_centers)
            ),
            dim=1
        )
        output = self.final_disc(output)
        return output

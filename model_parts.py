""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from new_afma.base._base import EncoderMixin

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Encoder_channelatt_img(nn.Module):
    def __init__(self, out_channels, classes_num=2, patch_size=8, depth=5, att_depth=1):
        super(Encoder_channelatt_img, self).__init__()
        self._depth = depth
        self._attention_on_depth = att_depth
        self._out_channels = out_channels
        self.patch_size = patch_size

        self.conv_img = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classes_num, kernel_size=3, padding=1)
        )

        self.layers = nn.ModuleList()
        for i in range(depth):
            in_ch = out_channels[i - 1] if i > 0 else 3
            out_ch = out_channels[i]
            layer = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)

        self.conv_feamap=nn.Sequential(
            nn.Conv2d(self._out_channels[self._attention_on_depth], classes_num, kernel_size=(1, 1), stride=1)
        )

        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))

        self.resolution_trans = nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size, 2 * self.patch_size * self.patch_size, bias=False),
            nn.Linear(2 * self.patch_size * self.patch_size, self.patch_size * self.patch_size, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        features = []
        attentions = []

        ini_img = self.conv_img(x)
        features.append(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            features.append(x)

            if i == self._attention_on_depth:
                feamap = self.conv_feamap(x) / (2 ** self._attention_on_depth * 2 ** self._attention_on_depth)
                for j in range(feamap.size(1)):
                    unfold_img = self.unfold(ini_img[:, j:j + 1, :, :]).transpose(-1, -2)
                    unfold_img = self.resolution_trans(unfold_img)

                    unfold_feamap = self.unfold(feamap[:, j:j + 1, :, :])
                    unfold_feamap = self.resolution_trans(unfold_feamap.transpose(-1, -2)).transpose(-1, -2)
                    att = torch.matmul(unfold_img, unfold_feamap) / (self.patch_size * self.patch_size)
                    attentions.append(torch.unsqueeze(att, 1))

                attentions = torch.cat(attentions, dim=1)


        return attentions

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, patch_size=8, activation=None, upsampling=1, att_depth=3):
        super().__init__()
        self.patch_size = patch_size
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.out_channels = out_channels
        self.upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=True) if upsampling > 1 else nn.Identity()
        self.unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        self.activation = activation if activation is not None else nn.Identity()
        self.att_depth = att_depth

    def forward(self, x, attentions):
        conv_feamap_size = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(2 ** self.att_depth, 2 ** self.att_depth),
                      stride=(2 ** self.att_depth, 2 ** self.att_depth), groups=self.out_channels, bias=False)
        conv_feamap_size.weight = nn.Parameter(torch.ones((self.out_channels, 1, 2 ** self.att_depth, 2 ** self.att_depth)))
        conv_feamap_size.to(x.device)
        for param in conv_feamap_size.parameters():
            param.requires_grad = False

        x = self.conv_x(x)
        x = self.upsampling(x)
        fold_layer = torch.nn.Fold(output_size=(x.size()[-2], x.size()[-1]), kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        correction = []

        x_argmax = torch.argmax(x, dim=1)
        print(f"After argmax shape: {x_argmax.shape}")

        pr_temp = torch.zeros(x.size()).to(x.device)
        src = torch.ones(x.size()).to(x.device)
        x_softmax = pr_temp.scatter(dim=1, index=x_argmax.unsqueeze(1), src=src)

        argx_feamap = conv_feamap_size(x_softmax) / (2 ** self.att_depth * 2 ** self.att_depth)

        for i in range(x.size()[1]):
            unfold_img = self.unfold(argx_feamap[:, i:i + 1, :, :])  # [1, 64, 256]
            unfold_img = unfold_img.transpose(-1, -2)  # 变为 [1, 256, 64]
            non_zeros = torch.unsqueeze(torch.count_nonzero(attentions[:, i:i + 1, :, :], dim=-1) + 0.00001, dim=-1)
            att = torch.matmul(attentions[:, i:i + 1, :, :] / non_zeros, unfold_img.transpose(-1, -2))
            att = torch.squeeze(att, dim=1)
            att = fold_layer(att.transpose(-1, -2))
            correction.append(att)

        correction = torch.cat(correction, dim=1)

        correction_conv = nn.Conv2d(in_channels=8, out_channels=x.size(1), kernel_size=1).to(x.device)
        correction = correction_conv(correction)

        x = correction * x + x

        x = self.activation(x)

        return x, attentions
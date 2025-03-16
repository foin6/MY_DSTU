import sys
sys.path.append('/home/zixuwang/MyProjs/Segmentation/MY_DSTU/')
sys.path.append('/home/zixuwang/MyProjs/Segmentation/MY_DSTU/Decoders')
sys.path.append('/home/zixuwang/MyProjs/Segmentation/MY_DSTU/VRWKV')
sys.path.append('/home/zixuwang/MyProjs/Segmentation/MY_DSTU/TIF')

import torch
import torch.nn as nn
import torch.nn.functional as F

from VRWKV.vrwkv_encoder2 import Encoder
from VRWKV.vrwkv_decoder import VRWKV_Decoder
from Decoders import Decoder
from ImgFusion.Fusion import Cross_Att

groups = 32
VRWKV_L_path= "/home/zixuwang/MyProjs/Segmentation/backup/pretrain_model/vrwkv_l_22kto1k_384.pth"
VRWKV_B_path = "/home/zixuwang/MyProjs/Segmentation/backup/pretrain_model/vrwkv_b_in1k_224.pth"
VRWKV_S_path = "/home/zixuwang/MyProjs/Segmentation/backup/pretrain_model/vrwkv_s_in1k_224.pth"
VRWKV_T_path = "/home/zixuwang/MyProjs/Segmentation/backup/pretrain_model/vrwkv_t_in1k_224.pth"

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), # 特征图的高宽缩小到原来的1/2
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True), # 不改变特征图的大小
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True), # 不改变特征图的大小
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True), # 特征图大小不变，只改变了通道数
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True), # 特征图大小不变，只改变了通道数
            nn.GroupNorm(num_channels=out_ch,num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x) # 只改变的通道数
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        #input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, dim, n_class, in_ch=3):
        super(UNet, self).__init__()
        # encoder 部分
        self.encoder = Encoder(patch_size=4, embed_dim=128, drop_path_rate=0.5, pretrained=None)
        # Decoder部分
        self.layer1 = Up(8*dim, 4*dim, bilinear=False)
        self.layer2 = Up(4*dim, 2*dim, bilinear=False)
        self.layer3 = Up(2*dim, dim, bilinear=False)
        self.layer4 = Decoder.Decoder(dim, dim, dim // 2)
        self.layer5 = Decoder.Decoder(dim // 2, dim // 2, dim // 4)
        # 其他部分
        self.down1 = nn.Conv2d(in_ch, dim // 4, kernel_size=1, stride=1, padding=0) # 不改变特征图大小
        self.down2 = conv_block(dim // 4, dim // 2) # 经过这层后特征图的高宽缩小到原来的1/2
        # 最终输出层
        self.final = nn.Conv2d(dim // 4, n_class, kernel_size=1, stride=1, padding=0)

        self.loss1 = nn.Sequential(
            nn.Conv2d(dim * 8, n_class, kernel_size=1, stride=1, padding=0), # 不改变特征图大小，只是将通道数变成1
            nn.ReLU(),
            nn.Upsample(scale_factor=32)
        )

        self.loss2 = nn.Sequential(
            nn.Conv2d(dim, n_class, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=4)
        )

    def forward(self, x): # x.shape=[batch_size, 3, H, W] # 直接输入的是原始图像
        out = self.encoder(x) # 这是patch_size是4×4、dim=128的那一层encoder
        e1, e2, e3, e4 = out[0], out[1], out[2], out[3]  # 4级编码器的输出
        loss1 = self.loss1(e4) # [batch_size, 1, Wh_e//8*32, Ww_e//8*32] # 图像的高宽变成原图的大小

        ds1 = self.down1(x) # [batch_size, model_dim//4, H, W] # 原图大小，维度从3变成model_dim/4
        ds2 = self.down2(ds1) # [batch_size, model_dim//2, H//2, W//2]

        # Decoder部分
        d1 = self.layer1(e4, e3) # [batch_size, model_dim*4, Wh_e//4, Ww_e//4]
        d2 = self.layer2(d1, e2) # [batch_size, model_dim*2, Wh_e//2, Ww_e//2]
        d3 = self.layer3(d2, e1) # [batch_size, model_dim, Wh_e, Ww_e]  Wh_e = H//4 Ww_e = W//4
        loss2 = self.loss2(d3) # [batch_size, 1, 4*Wh_e, 4*Ww_e] 变成一幅灰度图像，高宽与原图相同了
        d4 = self.layer4(d3, ds2) # [batch_size, model_dim//2, H//2, W//2]
        d5 = self.layer5(d4, ds1) # [batch_size, model_dim//4, H, W]
        o = self.final(d5) # 输出灰度图像 [batch_size, 1, H, W]
        return o, loss1, loss2 # shape全是[batch_size, 1, H, W] # 这三个输出在training中是要用于计算loss值的

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(1, 3, 384, 384)).cuda()
    model = UNet(128, 1).cuda()
    total_param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 模型中的全部参数量
    print("{0} parameters to be trained in total".format(total_param_num))  # 查看可训练的参数量
    print("Input shape:", x.shape)
    y = model(x)
    print('Output shape:',y[-1].shape)
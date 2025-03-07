import torch
import torch.nn as nn

from SwinTrans.SwinTransformer import SwinTransformer
from Decoders import Swin_Decoder, Decoder
from TIF import Cross_Att

groups = 32

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

class UNet(nn.Module):
    def __init__(self, dim, n_class, in_ch=3):
        super(UNet, self).__init__()
        # encoder 部分
        self.encoder = SwinTransformer(depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], drop_path_rate=0.5, embed_dim=128)
        self.encoder2 = SwinTransformer(depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], drop_path_rate=0.2, patch_size=8, embed_dim=96)
        self.encoder.init_weights('../DS-TransUNet/Swin-Transformer/swin_base_patch4_window7_224_22k.pth')
        self.encoder2.init_weights('../DS-TransUNet/Swin-Transformer/swin_tiny_patch4_window7_224.pth')
        # Decoder部分
        self.layer1 = Swin_Decoder.Swin_Decoder(8 * dim, 2, 8) # (in_channels, depths, num_heads) 1024
        self.layer2 = Swin_Decoder.Swin_Decoder(4 * dim, 2, 4) # 512
        self.layer3 = Swin_Decoder.Swin_Decoder(2 * dim, 2, 2) # 256
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
        dim_s = 96 # patch_size大的
        dim_l = 128 # patch_size小的
        self.m1 = nn.Upsample(scale_factor=2) # 将特征图的H和W扩展为原来的2倍，dim不变
        self.m2 = nn.Upsample(scale_factor=4)
        tb = dim_s + dim_l
        self.change1 = Conv_block(tb, dim)
        self.change2 = Conv_block(tb * 2, dim * 2)
        self.change3 = Conv_block(tb * 4, dim * 4)
        self.change4 = Conv_block(tb * 8, dim * 8)
        self.cross_att_1 = Cross_Att.Cross_Att(dim_s * 1, dim_l * 1)
        self.cross_att_2 = Cross_Att.Cross_Att(dim_s * 2, dim_l * 2)
        self.cross_att_3 = Cross_Att.Cross_Att(dim_s * 4, dim_l * 4)
        self.cross_att_4 = Cross_Att.Cross_Att(dim_s * 8, dim_l * 8)

    def forward(self, x): # x.shape=[batch_size, 3, H, W]
        # 以下两个encoder是处理不同scale图像的encoder
        out = self.encoder(x)
        out2 = self.encoder2(x)
        e1, e2, e3, e4 = out[0], out[1], out[2], out[3]  # 4级编码器的输出
        r1, r2, r3, r4 = out2[0], out2[1], out2[2], out2[3]
        # TIF的过程
        e1, r1 = self.cross_att_1(e1, r1) # [batch_size, dim_l, Wh_e, Ww_e]，[batch_size, dim_s, Wh_r, Ww_r]
        e2, r2 = self.cross_att_2(e2, r2) # [batch_size, dim_l*2, Wh_e//2, Ww_e//2]，[batch_size, dim_s*2, Wh_r//2, Ww_r//2]
        e3, r3 = self.cross_att_3(e3, r3) # [batch_size, dim_l*4, Wh_e//4, Ww_e//4]，[batch_size, dim_s*4, Wh_r//4, Ww_r//4]
        e4, r4 = self.cross_att_4(e4, r4) # [batch_size, dim_l*8, Wh_e//8, Ww_e//8]，[batch_size, dim_s*8, Wh_r//8, Ww_r//8]
        e1 = torch.cat([e1, self.m1(r1)], 1) # [batch_size, dim_l+dim_s, Wh_e, Ww_e]
        e2 = torch.cat([e2, self.m1(r2)], 1) # [batch_size, dim_l*2+dim_s*2, Wh_e//2, Ww_e//2]
        e3 = torch.cat([e3, self.m1(r3)], 1) # [batch_size, dim_l*4+dim_s*4, Wh_e//4, Ww_e//4]
        e4 = torch.cat([e4, self.m1(r4)], 1) # [batch_size, dim_l*8+dim_s*8, Wh_e//8, Ww_e//8]
        e1 = self.change1(e1) # [batch_size, model_dim, Wh_e, Ww_e]
        e2 = self.change2(e2) # [batch_size, model_dim*2, Wh_e//2, Ww_e//2]
        e3 = self.change3(e3) # [batch_size, model_dim*4, Wh_e//4, Ww_e//4]
        e4 = self.change4(e4) # [batch_size, model_dim*8, Wh_e//8, Ww_e//8]
        loss1 = self.loss1(e4) # [batch_size, 1, Wh_e//8*32, Ww_e//8*32] # 图像的高宽变成原图的大小

        ds1 = self.down1(x) # [batch_size, model_dim//4, H, W]
        ds2 = self.down2(ds1) # [batch_size, model_dim//2, H//2, W//2]

        # Decoder部分
        d1 = self.layer1(e4, e3) # [batch_size, model_dim*4, Wh_e//4, Ww_e//4]
        d2 = self.layer2(d1, e2) # [batch_size, model_dim*2, Wh_e//2, Ww_e//2]
        d3 = self.layer3(d2, e1) # [batch_size, model_dim, Wh_e, Ww_e]  Wh_e = H//4 Ww_e = W//4
        loss2 = self.loss2(d3) # [batch_size, 1, 4*Wh_e, 4*Ww_e] 变成一幅灰度图像，高宽与原图相同了
        d4 = self.layer4(d3, ds2) # [batch_size, model_dim//2, H//2, W//2]
        d5 = self.layer5(d4, ds1) # [batch_size, model_dim//4, H, W]
        o = self.final(d5) # 输出灰度图像 [batch_size, 1, H, W]
        return o, loss1, loss2 # shape全是[batch_size, 1, H, W]

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(1, 3, 512, 512)).cuda()
    model = UNet(128, 1).cuda()
    print("Input shape:", x.shape)
    y = model(x)
    print('Output shape:',y[-1].shape)
import sys
sys.path.append('/home/zixuwang/MyProjs/Segmentation/MY_DSTU/')
sys.path.append('/home/zixuwang/MyProjs/Segmentation/MY_DSTU/utils')
sys.path.append('/home/zixuwang/MyProjs/Segmentation/MY_DSTU/VRWKV')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import logging

from utils.checkpoint import load_checkpoint
########################################## 一些零件 ##########################################
from vrwkv import Block # 一个Block里面有1个Saptial Mix和1个Channel Mix
from vrwkv import VRWKV

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()
        patch_size = to_2tuple(patch_size) # (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 通过下面这个卷积网络后，图像的通道数变为embed_dim，图像的高宽变为原来的patch_size分之一
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x): # x.shape = [batch_size, 3, H, W]
        # padding 将图像填充成patch_size的整数倍，这样才能non-overlapped Patch Embedding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        # 开始进行Patch Embedding
        x = self.proj(x) # [batch_size, embed_dim, new_H, new_W]
        if self.norm is not None:
            new_H, new_W = x.shape[2], x.shape[3]
            x = x.flatten(2).transpose(1, 2) # LayerNorm是每个样本（patch）都要计算一遍所有特征的均值和方差，所以要faltten
            x = self.norm(x)
            # 恢复
            x = x.transpose(1, 2).view(-1, self.embed_dim, new_H, new_W) # [batch_size, embed_dim, new_H, new_W]
        return x # [batch_size, embed_dim, H_patch, W_patch]

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W): # [batch_size, patches_num, dim]
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C) # [batch_size, H, W, dim]

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x) # 在这里减少维度数

        return x  # [batch_size, H/2*W/2, 2*dim]
    
class FlexiblePatchMerging(nn.Module):
    def __init__(self, dim, downsample_rate, norm_layer=nn.LayerNorm):
        super().__init__()
        assert (downsample_rate & (downsample_rate - 1) == 0) and downsample_rate != 0, "downsample_rate must be a power of 2"
        self.dim = dim
        self.downsample_rate = downsample_rate
        self.reduction1 = nn.Linear(downsample_rate*dim, dim, bias=False)
        self.gelu = nn.GELU()
        self.reduction2 = nn.Linear(dim, downsample_rate*dim, bias=False)
        self.norm1 = norm_layer(downsample_rate*dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x):  # [batch_size, dim, H, W]
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)  # [batch_size, H, W, dim]

        # padding to ensure the image can be divided by the downsample rate
        pad_input = (H % self.downsample_rate != 0) or (W % self.downsample_rate != 0)
        if pad_input:
            h_pad = (self.downsample_rate - H % self.downsample_rate) % self.downsample_rate
            w_pad = (self.downsample_rate - W % self.downsample_rate) % self.downsample_rate
            x = F.pad(x, (0, 0, 0, w_pad, 0, h_pad))

        new_H, new_W = H // self.downsample_rate, W // self.downsample_rate
        patches = []
        for i in range(self.downsample_rate):
            for j in range(self.downsample_rate):
                patches.append(x[:, i::self.downsample_rate, j::self.downsample_rate, j*(C//self.downsample_rate):(j+1)*(C//self.downsample_rate)])
        x = torch.cat(patches, dim=-1)  # B, new_H, new_W, downsample_rate*C
        x = x.view(B, -1, self.downsample_rate*C)  # B, new_H*new_W, downsample_rate*C

        x = self.norm1(x) # [batch_size, new_H*new_W, downsample_rate*C]
        # Reduce dimensionality
        x = self.reduction1(x)
        x = self.gelu(x)
        x = self.norm2(x)
        x = self.reduction2(x).view(B, new_W, new_W, -1).permute(0, 3, 1, 2).contiguous()

        return x # [batch_size, dim, new_H, new_W]

########################################## 开始组装 ##########################################

class Encoder(nn.Module):
    def __init__(self, img_size=384, patch_size=4, in_chans=3, embed_dim=768,
                depth=12, drop_rate=0., drop_path_rate=0.5, out_indices=(1,5,9,11), 
                channel_gamma=1/4, shift_pixel=1, shift_mode="q_shift", 
                init_values=None, init_mode="fancy", post_norm=False, key_norm=False, 
                final_norm=True, hidden_rate=4, interpolate_mode='bicubic', 
                pretrained=None, with_cp=False, init_cfg=None):
        super(Encoder, self).__init__()

        self.out_indices = out_indices
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = VRWKV(img_size=img_size, patch_size=patch_size, in_channels=in_chans, 
                            out_indices=out_indices, drop_rate=drop_rate, embed_dims=embed_dim,
                            depth=depth, drop_path_rate=drop_path_rate, channel_gamma=channel_gamma,
                            shift_pixel=shift_pixel, init_values=init_values, shift_mode=shift_mode,
                            init_mode=init_mode, post_norm=post_norm, key_norm=key_norm, hidden_rate=hidden_rate,
                            final_norm=final_norm, interpolate_mode=interpolate_mode, pretrained=pretrained,
                            with_cp=with_cp, init_cfg=init_cfg)
        self.merges = nn.ModuleList()
        for i in range(len(out_indices)-1):
            merge_layer = FlexiblePatchMerging(dim=embed_dim, downsample_rate=2**(i+1))
            self.merges.append(merge_layer)

    def forward(self, x): # [batch_size, 3, H, W]
        self.encoder.init_weights() # 加载预训练模型
        outs = list(self.encoder(x))
        for i in range(len(self.out_indices)-1):
            outs[i+1] = self.merges[i](outs[i+1])
        return outs # 用于skip connection

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(Encoder, self).train(mode)


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(1, 3, 384, 384)).cuda()
    encoder = Encoder(img_size=384, patch_size=4, in_chans=3, embed_dim=96,
                    depth=12, drop_rate=0., drop_path_rate=0.5, out_indices=(1,5,9,11), 
                    channel_gamma=1/4, shift_pixel=1, shift_mode="q_shift", 
                    init_values=None, init_mode="fancy", post_norm=False, key_norm=False, 
                    final_norm=True, hidden_rate=4, interpolate_mode='bicubic', 
                    pretrained="/home/zixuwang/MyProjs/Segmentation/backup/pretrain_model/vrwkv_b_in1k_224.pth", 
                    with_cp=False, init_cfg=None).cuda()
    total_param_num = sum(p.numel() for p in encoder.parameters() if p.requires_grad)  # 模型中的全部参数量
    print("{0} parameters to be trained in total".format(total_param_num))  # 查看可训练的参数量
    print("Input shape:", x.shape)
    out = encoder(x)
    print('Output shape:{} {} {} {}'.format(out[0].shape, out[1].shape, out[2].shape, out[3].shape))
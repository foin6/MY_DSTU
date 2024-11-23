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
        x = self.reduction(x)

        return x  # [batch_size, H/2*W/2, 2*dim]

########################################## 开始组装 ##########################################
class BasicLayer(nn.Module): # 这就是一个stage
    def __init__(self, use_checkpoint, downsample, norm_layer, drop_path_rate,
                 embed_dims=256, depth=12, channel_gamma=1/4, shift_pixel=1, 
                 init_values=None, shift_mode='q_shift', init_mode='fancy',
                 post_norm=False, key_norm=False, hidden_rate=4, with_cp=False, up=True):
        super(BasicLayer, self).__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.up = up

        # build blocks
        dpr = drop_path_rate # dropout rate是变化的,是一个列表
        self.blocks = nn.ModuleList([
            Block(
                n_embd=embed_dims,
                n_layer=depth,
                layer_id=i, # 这是第几个layer
                channel_gamma=channel_gamma, # 1/4
                shift_pixel=shift_pixel, # 1
                shift_mode=shift_mode, # q_shift
                hidden_rate=hidden_rate, # 4
                drop_path=dpr[i],
                init_mode=init_mode, # fancy
                init_values=init_values, # None
                post_norm=post_norm, # False
                key_norm=key_norm, # False
                with_cp=with_cp # False
            ) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=embed_dims, norm_layer=norm_layer) # 长宽变成原来的一般，维度是原来的两倍
        else:
            self.downsample = None

    def forward(self, x, H, W): # [batch_size, total_patches_num, embed_dim]
        patch_resolution = (H, W)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, patch_resolution)
            else:
                x = blk(x, patch_resolution)  # [batch_size, patches_num, dim]
        # satge的最后一个部分是PatchMerge：feature map的各个维度数减半，通道数翻倍
        if self.downsample is not None:
            x_down = self.downsample(x, H, W) # [batch_size, H/2*W/2, 2*dim]
            if self.up: # 关系到UNet上采样的部分
                Wh, Ww = (H+1)//2, (W+1)//2
            else:
                Wh, Ww = H*2, W*2
            return x, H, W, x_down, Wh, Ww
        else: # encoder的最后一个stage不需要PatchMerge，encode也都走这个判断
            return x, H, W, x, H, W


class Encoder(nn.Module):
    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=128,
                 depths=[2,2,18,2], drop_rate=0., drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0,1,2,3),
                 frozen_stages=-1, use_checkpoint=False, channel_gamma=1/4, shift_pixel=1,
                 shift_mode="q_shift", init_values=None, init_mode="fancy", post_norm=False,
                 key_norm=False, hidden_rate=4, with_cp=False):
        super(Encoder, self).__init__()
        self.pretrain_img_size = pretrain_img_size # 预训练时使用的image size
        self.num_layers = len(depths) # 表示整个Encoder有几个完整的Stage，depths[i]表示第i个stage中有depths[i]个vrwkv block
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # 将图像以non-overlap的方式分成patches
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:  # 默认是False
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])) # 这个patches_resolution是要调整的
            trunc_normal_(self.absolute_pos_embed, std=.02) # 设定参数符合正态分布

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # 将drop_path_rate平均切分（24份）

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 一个layer是一个stage，每个stage中包含多个VRWKV Block
            layer = BasicLayer( # 这就是一个stage
                embed_dims=int(embed_dim * 2 ** i_layer), # 越深处的stage模型的维度越大，因为经过Patch Merging之后特征图会逐渐变小，增加维度是一种补偿方式
                depth=depths[i_layer], # 在第i个stage中共有depths[i_layer]个VRWKV Block
                drop_path_rate=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], # 取出对应于当前stage的那几份drop_path_rate
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,  # 最后一个stage不要PatchMerging
                use_checkpoint=use_checkpoint,
                channel_gamma=channel_gamma, # 1/4
                shift_pixel=shift_pixel, # 1
                shift_mode=shift_mode, # q_shift
                init_values=init_values, # None
                init_mode=init_mode, # fancy
                post_norm=post_norm, # False
                key_norm=key_norm, # False
                hidden_rate=hidden_rate, # 4
                with_cp=with_cp # False
            )
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)] # 不同stage中的模型维度数不同
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices: # 需要有输出的这些stages后面都加上layerNorm
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages() # 设置了frozen_stages=-1，即不进行冻结操作

    def _freeze_stages(self): # 冻结模型的特定部分，设置一些层的参数为不需要梯度，这样在模型训练时这些参数就不会被更新（适用于微调的过程）
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x): # [batch_size, 3, H, W]
        x = self.patch_embed(x) # [batch_size, embed_dim, H_patch, W_patch]

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:  # 绝对位置编码 # 默认是False
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic') # 使用双三次插值调整self.absolute_pos_embed的shape，使其能与x相加
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # [batch_size, total_patches_num, embed_dim]
        else:
            x = x.flatten(2).transpose(1, 2) # [batch_size, total_patches_num, embed_dim] # 进入Swin前进行shape的转变
        x = self.pos_drop(x) # dropout层

        outs = []
        for i in range(self.num_layers): # 遍历每个stage
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww) # 通过一个stage的处理
            if i in self.out_indices: # 第i个stage的输出是要用于skip connection的
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out) # 输出进行一次LayerNorm的操作

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs # 收集起来，用于skip connection

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(Encoder, self).train(mode)
        self._freeze_stages()


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(1, 3, 512, 512)).cuda()
    # basicLayer = BasicLayer(use_checkpoint=False, downsample=PatchMerging, norm_layer=nn.LayerNorm,
    #              embed_dims=256, depth=12, drop_path_rate=[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #              channel_gamma=1/4, shift_pixel=1, init_values=None,
    #              shift_mode='q_shift', init_mode='fancy',
    #              post_norm=False, key_norm=False,
    #              hidden_rate=4, with_cp=False, up=True).cuda()
    encoder = Encoder(pretrain_img_size=224, patch_size=8, in_chans=3, embed_dim=96,
                 depths=[2,2,6,2], drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0,1,2,3),
                 frozen_stages=-1, use_checkpoint=False, channel_gamma=1/4, shift_pixel=1,
                 shift_mode="q_shift", init_values=None, init_mode="fancy", post_norm=False,
                 key_norm=False, hidden_rate=4, with_cp=False).cuda()
    total_param_num = sum(p.numel() for p in encoder.parameters() if p.requires_grad)  # 模型中的全部参数量
    print("{0} parameters to be trained in total".format(total_param_num))  # 查看可训练的参数量
    print("Input shape:", x.shape)
    out= encoder(x)
    print('Output shape:{} {} {} {}'.format(out[0].shape, out[1].shape, out[2].shape, out[3].shape))
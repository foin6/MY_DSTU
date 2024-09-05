import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import logging

from utils.checkpoint import load_checkpoint
########################################## 一些零件 ##########################################
from SwinTrans.SwinTrans_utils.window_partition import window_partition
from SwinTrans.SwinTrans_utils.window_reverse import window_reverse

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

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features).cuda()
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features).cuda()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置编码表
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02) # 初始化一个正态分布

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.meshgrid([coords_h, coords_w]) # 获得所有像素点的坐标位置 # 返回一个元组（x轴坐标，y轴坐标）
        coords = torch.stack(coords) # 将x轴坐标和y轴坐标进行堆叠 [2, win_H, win_W]
        coords_flatten = torch.flatten(coords, 1)  # 将坐标拉直 [2, win_H*win_W]，形成坐标
        # 计算坐标的相对位置（采用广播机制）
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww # 每个横坐标-所有横坐标  每个纵坐标-所有纵坐标
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # 绝对坐标的编码过程
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 所有相对横坐标从0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 所有相对纵坐标从0开始
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # 所有横坐标*(2*win_W-1)，使得同一行的不同像素间横纵坐标相加的值不同
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww # 由此得到的还是相对位置偏移的索引值
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1) # 在行的维度上进行softmax

    def forward(self, x, mask): # [batch_size*total_win_num, patches_num_per_window, dim]
        B_, N, C = x.shape # [batch_size*total_win_num, patches_num_per_window, dim]
        # qkv(x) ---> [batch_size*total_win_num, patches_num_per_window, 3*embed_dim]
        # reshape ---> [batch_size*total_win_num, patches_num_per_window, 3, heads_num, head_dim]
        # permute ---> [3, batch_size*total_win_num, heads_num, patches_num_per_window, head_dim] # 为窗口内进行attention操作做准备
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [batch_size*total_win_num, heads_num, patches_num_per_window, patches_num_per_window]

        # 加上Relative Position Bias（查表获得）
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # [patches_num_per_window, patches_num_per_window, heads_num]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # [heads_num, patches_num_per_window, patches_num_per_window]
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:  # 这个分支是进行 Shift Window MSA，还要加上一个mask
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N) # [batch_size*total_win_num, heads_num, patches_num_per_window, patches_num_per_window]
            attn = self.softmax(attn) # [batch_size*total_win_num, heads_num, patches_num_per_window, patches_num_per_window]
        else:  # 这个分支是直接进行MSA
            attn = self.softmax(attn) # [batch_size*total_win_num, heads_num, patches_num_per_window, patches_num_per_window]

        attn = self.attn_drop(attn) # [batch_size*total_win_num, heads_num, patches_num_per_window, patches_num_per_window]

        # 以下步骤是进行多头合成
        # @ ---> [batch_size*total_win_num, heads_num, patches_num_per_window, head_dim]
        # transpose ---> [batch_size*total_win_num, patches_num_per_window, heads_num, head_dim]
        # reshape ---> [batch_size*total_win_num, patches_num_per_window, dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)  # Wo # [batch_size*total_win_num, patches_num_per_window, dim]
        x = self.proj_drop(x)
        return x  # [batch_size*total_win_num, patches_num_per_window, dim]



########################################## 开始组装 ##########################################
class SwinTransformerBlock(nn.Module): # 一个W-MSA或SW-MSA
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio, qkv_bias,
                 qk_scale, drop, attn_drop, drop_path, norm_layer, act_layer=nn.GELU):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix): # x.shape=[batch_size, total_patches_num, dim], mask_matrix.shape=# [total_win_num, patches_num_per_window, patches_num_per_window]
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x # [batch_size, total_patches_num, dim]
        x = self.norm1(x) # [batch_size, total_patches_num, dim]
        x = x.view(B, H, W, C) # [batch_size, H_patch, W_patch, dim]

        # pad feature maps to multiples of window size(图像填充，将feature map的各个维度都填充成window_size的整数倍)
        # 为了之后方便还原，在右边、下边进行填充
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape # [batch_size, Hp, Wp, dim]

        # cyclic shift
        if self.shift_size > 0:  # SW-MSA首先需要进行窗口滚动
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) # [batch_size, Hp, Wp, dim]
            attn_mask = mask_matrix
        else:  # W-MSA不需要进行窗口滚动
            shifted_x = x # [batch_size, Hp, Wp, dim]
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [batch_size*total_windows_num, win_size_H, win_size_W, dim]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [batch_size*total_win_num, patches_num_per_window, dim]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [batch_size*total_win_num, patches_num_per_window, dim]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [batch_size*total_windows_num, win_size_H, win_size_W, dim]
        # 下面这步是去除窗口划分，全部重新变成填充后的图像大小
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [batch_size, Hp, Wp, dim]

        # reverse cyclic shift
        if self.shift_size > 0:  # 进行过shift window操作，说明此时进行的步骤是SW-MSA，那么还要给他翻转回来
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) # [batch_size, Hp, Wp, dim]
        else:  # 没有进行过shift window操作，说明此时进行的步骤是W-MSA
            x = shifted_x # [batch_size, Hp, Wp, dim]

        # 如果有过填充操作，那么也要删除之前的填充部分，这样才能进行残差连接
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous() # [batch_size, H_patch, W_patch, dim]

        x = x.view(B, H * W, C)  # [batch_size, total_patches_num, dim]

        # FFN
        x = shortcut + self.drop_path(x)  # [batch_size, total_patches_num, dim]
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x  # [batch_size, total_patches_num, embed_dim]


class BasicLayer(nn.Module): # 这就是一个stage
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
                 drop, attn_drop, drop_path, norm_layer, downsample, use_checkpoint, up=True):
        super(BasicLayer, self).__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 # SW-MSA中的窗口偏移量
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.up = up

        # build blocks # 每个block是一个W-MSA或SW-MSA(用i的奇偶性来确定到底是那个)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size = 0 if (i % 2 == 0) else window_size // 2, # 以此来区分是否要进行窗口的偏移，每次的偏移量是窗口大小的一半
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, # 每个block使用的drop是不一样的，这个在之前已经分好了
                norm_layer=norm_layer
            ) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer) # 长宽变成原来的一般，维度是原来的两倍
        else:
            self.downsample = None

    def forward(self, x, H, W): # [batch_size, total_patches_num, embed_dim]
        #  calculate attention mask for SW-MSA
        # 将特征图的Height和Width填充成window size大小的整数倍，以便能分成整数个窗口
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1 Hp Wp 1]
        # 分区，shifted window之后被分成了9个区
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        # 分区编号（0-8） # 只需要9个编号即可
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # 将img_mask分成一个个互不重叠的窗口windows
        mask_windows = window_partition(img_mask, self.window_size)  # [total_windows_num, win_size_H, win_size_W, 1]
        # 下面这一步骤生成的二维矩阵中，每一行代表一个窗口中的mask值
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [total_win_num, patches_num_per_window]
        # 下面使用广播机制来生成mask，保证只有处于相同区内的元素进行attention操作才是有效的
        # unsqueeze(1) ---> [total_win_num, 1, patches_num_per_window]
        # unsqueeze(2) ---> [total_win_num, patches_num_per_window, 1]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [total_win_num, patches_num_per_window, patches_num_per_window]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))  # 在原本不在同一窗口处的地方添加上-100作为mask，这样经过softmax操作之后此处就接近于0了

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)  # [batch_size, patches_num, dim]
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


class SwinTransformer(nn.Module):
    def __init__(self, pretrain_img_size=224, patch_size=4, in_chans=3, embed_dim=128,
                 depths=[2,2,18,2], num_heads=[4,8,16,32], window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0,1,2,3),
                 frozen_stages=-1, use_checkpoint=False):
        super(SwinTransformer, self).__init__()
        self.pretrain_img_size = pretrain_img_size # 预训练时使用的image size
        self.num_layers = len(depths) # 表示整个SwinTransformer模型有几个完整的stage，depths[i]表示第i个stage中W-MSA和SW-MSA一共有depths[i]个
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
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # 将drop_path_rate分成24份

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 一个layer是一个stage，每个stage中包含多个W-MSA和SW-MSA
            layer = BasicLayer( # 这就是一个stage
                dim=int(embed_dim * 2 ** i_layer), # 越深处的stage模型的维度越大，因为经过Patch Merging之后特征图会逐渐变小，增加维度是一种补偿方式
                depth=depths[i_layer], # 在第i个stage中W-MSA和SW-MSA共有depths[i_layer]个
                num_heads = num_heads[i_layer], # 第i个stage中的多头数量
                window_size = window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], # 取出对应于当前stage的那几份drop_path_rate
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,  # 最后一个stage不要PatchMerging
                use_checkpoint=use_checkpoint
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
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


if __name__ == '__main__':
    x = torch.randn(size=(1, 3, 512, 512)).cuda()
    SwinTrans = SwinTransformer()
    out = SwinTrans(x)
    print(out.shape)
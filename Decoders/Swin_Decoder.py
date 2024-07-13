import torch
import torch.nn as nn

from Decoders.De_utils.up_conv import up_conv
from SwinTrans.SwinTransformer import BasicLayer

groups = 32

class SwinDecoder(nn.Module): # 这就是Decoder部分的一个stage
    def __init__(self, embed_dim, patch_size=4, depths=2, num_heads=6, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False):
        super(SwinDecoder, self).__init__()
        self.patch_norm = patch_norm
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # build layers
        self.layer = BasicLayer(
            dim=embed_dim // 2,
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr,
            norm_layer=norm_layer,
            downsample=None, # 不进行PatchMerge了
            use_checkpoint=use_checkpoint
        )
        self.up = up_conv(embed_dim, embed_dim // 2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x): # [batch_size, dim_x1, H_x1, W_x1]
        B, C, H, W = x.shape
        x = self.up(x)  # B , C//2, 2H, 2W # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
        x = x.reshape(B, C // 2, H * W * 4) # [batch_size, dim_x1//2, 4*H_x1*W_x1]
        x = x.permute(0, 2, 1) # [batch_size, 4*H_x1*W_x1, dim_x1//2]

        x_out, H, W, x, Wh, Ww = self.layer(x, H * 2, W * 2) # x.shape=[batch_size, 4*H_x1*W_x1, dim_x1//2]  x_out.shape=[batch_size, 4*H_x1*W_x1, dim_x1//2]

        x = x.permute(0, 2, 1) # [batch_size, dim_x1//2, 4*H_x1*W_x1]
        x = x.reshape(B, C // 2, H, W) # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
        # B, C//4 2H, 2W
        x = self.conv_relu(x) # [batch_size, dim_x1//4, 2*H_x1, 2*W_x1]

        return x # [batch_size, dim_x1//4, 2*H_x1, 2*W_x1]


class Swin_Decoder(nn.Module):
    def __init__(self, in_channels, depths, num_heads):
        super(Swin_Decoder, self).__init__()
        self.up = SwinDecoder(in_channels, depths=depths, num_heads=num_heads)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x1, x2): # x1.shape = [batch_size, dim_x1, H_x1, W_x1]   x2.shape=[batch_size, dim_x2, H_x2, W_x2]  (dim_x2=dim_x1//2, in_channels=dim_x1)
        x1 = self.up(x1) # [batch_size, dim_x1//4, 2*H_x1, 2*W_x1]
        x2 = self.conv2(x2) # [batch_size, dim_x2//2, H_x2, W_x2] = [batch_size, dim_x1//4, 2*H_x1, 2*W_x1]
        x1 = torch.cat((x2, x1), dim=1) # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
        out = self.conv_relu(x1) # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
        return out # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]

import torch
import torch.nn as nn

from utils.module import Attention, PreNorm, FeedForward

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x): # [batch_size, total_patches_num_r+1, dim_s] or [batch_size, total_patches_num_e+1, dim_l]
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Cross_Att(nn.Module):
    def __init__(self, dim_s, dim_l):
        super().__init__()
        self.transformer_s = Transformer(dim=dim_s, depth=1, heads=3, dim_head=32, mlp_dim=128)
        self.transformer_l = Transformer(dim=dim_l, depth=1, heads=1, dim_head=64, mlp_dim=256)
        self.norm_s = nn.LayerNorm(dim_s)
        self.norm_l = nn.LayerNorm(dim_l)
        self.avgpool = nn.AdaptiveAvgPool1d(1) # 池化层，将每个通道的宽度缩减到1
        # 以下两个线性层用于维度互换
        self.linear_s = nn.Linear(dim_s, dim_l) 
        self.linear_l = nn.Linear(dim_l, dim_s)

    def forward(self, e, r):
       b_e, c_e, h_e, w_e = e.shape # [batch_size, dim_l, Wh_e, Ww_e]
       e = e.reshape(b_e, c_e, -1).permute(0, 2, 1) # [batch_size, total_patches_num_e, dim_l]
       b_r, c_r, h_r, w_r = r.shape # [batch_size, dim_s, Wh_r, Ww_r]
       r = r.reshape(b_r, c_r, -1).permute(0, 2, 1) # [batch_size, total_patches_num_r, dim_s]
       # transpose ---> [batch_size, dim, total_patches_num]
       # avgpool ---> [batch_size, dim, 1] # 池化层一个值
       # flatten ---> [batch_size, dim]
       e_t = torch.flatten(self.avgpool(self.norm_l(e).transpose(1,2)), 1) # [batch_size, dim_l]
       r_t = torch.flatten(self.avgpool(self.norm_s(r).transpose(1,2)), 1) # [batch_size, dim_s]
       # 接下来这两步是维度互换，以便于拼接到对方的特征图上，然后在Transformer操作中进行信息的互通
       e_t = self.linear_l(e_t).unsqueeze(1) # [batch_size, 1, dim_s]
       r_t = self.linear_s(r_t).unsqueeze(1) # [batch_size, 1, dim_l]
       # cat ---> [batch_size, total_patches_num_r+1, dim_s]
       r = self.transformer_s(torch.cat([e_t, r],dim=1))[:, 1:, :] # [batch_size, total_patches_num_r, dim_s] (有切片操作，还原成原图大小)
       # cat ---> [batch_size, total_patches_num_e+1, dim_l]
       e = self.transformer_l(torch.cat([r_t, e],dim=1))[:, 1:, :] # [batch_size, total_patches_num_e, dim_l] (有切片操作，还原成原图大小)
       e = e.permute(0, 2, 1).reshape(b_e, c_e, h_e, w_e) # [batch_size, dim_l, Wh_e, Ww_e]
       r = r.permute(0, 2, 1).reshape(b_r, c_r, h_r, w_r) # [batch_size, dim_s, Wh_r, Ww_r]
       return e, r
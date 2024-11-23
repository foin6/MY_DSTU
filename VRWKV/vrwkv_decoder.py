import sys
sys.path.append('/home/zixuwang/MyProjs/Segmentation/MY_DSTU/')
sys.path.append('/home/zixuwang/MyProjs/Segmentation/MY_DSTU/Decoders')
sys.path.append('/home/zixuwang/MyProjs/Segmentation/MY_DSTU/VRWKV')

import torch
import torch.nn as nn

from Decoders.De_utils.up_conv import up_conv
from vrwkv_encoder import BasicLayer

class DecoderBasicLayer(nn.Module): # 这就是Decoder部分的一个stage
    def __init__(self, use_checkpoint=False, norm_layer=nn.LayerNorm, drop_rate=0.,
                drop_path_rate=0.2, embed_dims=256, depths=2, channel_gamma=1/4, 
                shift_pixel=1, init_values=None, shift_mode='q_shift', init_mode='fancy',
                post_norm=False, key_norm=False, hidden_rate=4, with_cp=False):
        super(DecoderBasicLayer, self).__init__()

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        # build layers
        self.layer = BasicLayer(
            embed_dims=embed_dims//2, # embed_dims是输入的，经过upSampling之后会减半，因此BasicLayer处理的dim大小是embed_dims//2
            depth=depths,
            drop_path_rate=dpr,
            norm_layer=norm_layer,
            downsample=None, # 不进行PatchMerge了
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
        self.up = up_conv(embed_dims, embed_dims // 2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(embed_dims // 2, embed_dims // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x): # [batch_size, dim_x1, H_x1, W_x1]
        B, C, H, W = x.shape
        # UpSampling
        x = self.up(x)  # B , C//2, 2H, 2W # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
        x = x.reshape(B, C // 2, H * W * 4) # [batch_size, dim_x1//2, 4*H_x1*W_x1]
        x = x.permute(0, 2, 1) # [batch_size, 4*H_x1*W_x1, dim_x1//2]
        
        # BasicLayer
        x_out, H, W, x, Wh, Ww = self.layer(x, H * 2, W * 2) # x.shape=[batch_size, 4*H_x1*W_x1, dim_x1//2]  x_out.shape=[batch_size, 4*H_x1*W_x1, dim_x1//2]

        x = x.permute(0, 2, 1) # [batch_size, dim_x1//2, 4*H_x1*W_x1]
        x = x.reshape(B, C // 2, H, W) # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
        x = self.conv_relu(x) # [batch_size, dim_x1//4, 2*H_x1, 2*W_x1]

        return x # [batch_size, dim_x1//4, 2*H_x1, 2*W_x1]


class VRWKV_Decoder(nn.Module): # 这里是将residual_connection传来的内容也输入到stage中处理了，还是一个stage
    def __init__(self, this_embed_dims, depths):
        super(VRWKV_Decoder, self).__init__()
        self.up = DecoderBasicLayer(embed_dims=this_embed_dims, depths=depths)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(this_embed_dims//2, this_embed_dims//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(this_embed_dims//2, this_embed_dims//4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x1, x2): # x1.shape = [batch_size, dim_x1, H_x1, W_x1]   x2.shape=[batch_size, dim_x2, H_x2, W_x2]  (dim_x2=dim_x1//2, in_channels=dim_x1)
        x1 = self.up(x1) # [batch_size, dim_x1//4, 2*H_x1, 2*W_x1]
        x2 = self.conv2(x2) # [batch_size, dim_x2//2, H_x2, W_x2] = [batch_size, dim_x1//4, 2*H_x1, 2*W_x1]
        x1 = torch.cat((x2, x1), dim=1) # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
        out = self.conv_relu(x1) # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
        return out # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
    

if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x1 = Variable(torch.rand(1, 2048, 16, 16)).cuda()
    x2 = Variable(torch.rand(1, 1024, 32, 32)).cuda()
    encoder = VRWKV_Decoder(2048, 2).cuda()
    total_param_num = sum(p.numel() for p in encoder.parameters() if p.requires_grad)  # 模型中的全部参数量
    print("{0} parameters to be trained in total".format(total_param_num))  # 查看可训练的参数量
    print("Input shape:{} {}".format(x1.shape, x2.shape))
    out= encoder(x1, x2)
    print('Output shape:{}'.format(out.shape))
    


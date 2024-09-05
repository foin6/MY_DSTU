import torch.nn as nn

groups = 32

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True), # out_ch = in_ch // 2
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): # [batch_size, dim_x, H_x, W_x]
        x = self.up(x) # [batch_size, dim_x//2, 2*H_x, 2*W_x]
        return x
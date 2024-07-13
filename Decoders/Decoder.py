import torch
import torch.nn as nn

from Decoders.De_utils.up_conv import up_conv

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = up_conv(in_channels, out_channels)
    #self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        #coorAtt(out_channels),
        nn.ReLU(inplace=True)
        )

  def forward(self, x1, x2):
    x1 = self.up(x1) # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]   dim_x1//2 = dim_x
    x1 = torch.cat((x2, x1), dim=1) # [batch_size, dim_x1, 2*H_x1, 2*W_x1]
    x1 = self.conv_relu(x1) # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
    return x1 # [batch_size, dim_x1//2, 2*H_x1, 2*W_x1]
from torch.autograd import Variable, Function
import torch
from torch import nn
import numpy as np

class DConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=None):
        super(DConv2d, self).__init__()
        self.kernel_size  = kernel_size
        self.padding      = padding
        self.zero_padding = nn.ZeroPad2d(padding)
        self.offsets      = nn.Conv2d(in_channels,1, kernel_size=kernel_size, stride=1, bias=bias)
        self.conv_kernel  = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                      stride=kernel_size, bias=bias)
    def forward(self, x):
        offset = self.offsets(x)
        if self.padding:
            print('aa')
        
        x = self.zero_padding(x)
        
        
        return x
    
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        #x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        #return x_offset
        
        

Net = DConv2d(2, 1)

#input_ = torch.ones((1,2,3,3))
#input_ = Net(input_)
#print(input_)

input_ = torch.FloatTensor([[[[1,2,3],
                            [4,5,6],
                            [7,8,9]]]])
ks = 3
input_ = torch.cat([input_[..., s:s+ks].contiguous().view(1,1, 3, 3*ks) for s in range(0, 3, ks)], dim=-1)

print(input_.shape)
from models.yolo import Model
from utils.utils import *
import torch.nn as nn
import torch
import torch.nn.functional as F

class V1(nn.Module):
    def __init__(self):
        super(V1, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Upsample(size=500, mode='nearest')
        pass

    def forward(self, x):
        x = self.up(x)
        return x

class V2(nn.Module):
    def __init__(self):
        super(V2, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='nearest')
        pass

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(2.0, 2.0, ))
        return x

class V3(nn.Module):
    def __init__(self):
        super(V3, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='nearest')
        pass

    def forward(self, x):
        x = torch._C._nn.upsample_nearest2d(x, [500,500])
        return x

class V4(nn.Module):
    def __init__(self):
        super(V4, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='nearest')
        pass

    def forward(self, x):
        x = F.interpolate(x, size=(500,500,))
        return x

v1 = V1()
trace_model = torch.jit.trace(v1, (torch.Tensor(1,3,224,224), ))
trace_model.save('./weights/upsample_jit_v1.pt')

v2 = V2()
trace_model = torch.jit.trace(v2, (torch.Tensor(1,3,224,224), ))
trace_model.save('./weights/upsample_jit_v2.pt')

v3 = V3()
trace_model = torch.jit.trace(v3, (torch.Tensor(1,3,224,224), ))
trace_model.save('./weights/upsample_jit_v3.pt')

v4 = V4()
trace_model = torch.jit.trace(v4, (torch.Tensor(1,3,224,224), ))
trace_model.save('./weights/upsample_jit_v4.pt')

print('done')
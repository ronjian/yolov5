#%%
import numpy as np
import torch
import torch.nn as nn

# arr = np.load("./assets/output_torch/1.npy")
# arr = arr[0]
# # arr = np.reshape(arr, (28, -1, 64, 84))
# arr = np.float32(arr)
# input_data = torch.Tensor(arr)
# input_data.size()

# %%
class PostProcess(nn.Module):
    def __init__(self):
        super(PostProcess, self).__init__()

    def forward(self, x):
        # x = x[:,:,4:6] # slice
        # x = x > 0.0 # compare
        # x = x * 1.0
        # x = x + 1.0
        # x = x - 1.0
        # x = x / 0.01
        x = torch.gt(x, 0.0)
        return x

# %%
net = PostProcess()
_ = net(torch.Tensor(3, 224, 224))

# %%
trace_model = torch.jit.trace(net, (torch.Tensor(3, 224, 224), ))
trace_model.save('./weights/gt.jit')

# %%

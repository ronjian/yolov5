#%%
import numpy as np

rk1 = np.load("./assets/output_rk/1.npy")
rk2 = np.load("./assets/output_rk/2.npy")
rk3 = np.load("./assets/output_rk/3.npy")
rkinput = np.load("./assets/output_rk/input.npy")
torch1 = np.load("./assets/output_torch/1.npy")
torch2 = np.load("./assets/output_torch/2.npy")
torch3 = np.load("./assets/output_torch/3.npy")
torchinput = np.load("./assets/output_torch/input.npy")

#%%
rkinput.shape, torchinput.shape

# %%
np.max(rkinput), np.max(torchinput), np.min(rkinput), np.min(torchinput)

# %%

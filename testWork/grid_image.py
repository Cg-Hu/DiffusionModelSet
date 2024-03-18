from PIL import Image
import torch
from torchvision.utils import make_grid
from einops import rearrange
import numpy as np

A1 = np.array(Image.open("./data/A1.jpg"))
A2 = np.array(Image.open("./data/A2.jpg"))
B1 = np.array(Image.open("./data/B1.jpg"))
B2 = np.array(Image.open("./data/B2.jpg"))
k1 = np.array(Image.open("./data/k1.jpg"))
k2 = np.array(Image.open("./data/k2.jpg"))
A1 = torch.tensor(A1).permute(2, 0, 1)[None,:,:,:]
A2 = torch.tensor(A2).permute(2, 0, 1)[None,:,:,:]
B1 = torch.tensor(B1).permute(2, 0, 1)[None,:,:,:]
B2 = torch.tensor(B2).permute(2, 0, 1)[None,:,:,:]
k1 = torch.tensor(k1).permute(2, 0, 1)[None,:,:,:]
k2 = torch.tensor(k2).permute(2, 0, 1)[None,:,:,:]
print(A1.shape)

# l1 = []
# l1.append(A1)
# l1.append(A2)
# l2 = []
# l2.append(B1)
# l2.append(B2)
# l3 = []
# l3.append(k1)
# l3.append(k2)

A = torch.cat([A1, B1, k1], dim=0)
B = torch.cat([A2, B2, k2], dim=0)
print(A.shape)
# exit()
all_samples = []
all_samples.append(A)
all_samples.append(B)
# all_samples.append(k)
# 是这样的
# additionally, save as grid
grid = torch.stack(all_samples, 0)
grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=3)

# to image
grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
Image.fromarray(grid.astype(np.uint8)).save('result.png')

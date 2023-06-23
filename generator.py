import torch
import torch.nn as nn
from Impasto_diffusion_pytorch import Unet
model = Unet(dim=64, out_dim=3, with_time_emb=False, residual=False)
state_dict = torch.load("results/edge1/model.pt")

# Remove the "module." prefix from the keys if using DataParallel
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# Filter out unexpected or missing keys
state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}

# Load the filtered state_dict
model.load_state_dict(state_dict)
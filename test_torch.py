import torch
model = torch.jit.load("midas_small.torchscript")
print("✅ Loaded TorchScript model")
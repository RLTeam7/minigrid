import torch
def preprocess(obs,device):
    # Normalize pixel values and add a batch dimension (BCHW)
    return torch.tensor(obs['image'].transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
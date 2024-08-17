import torch

def load_checkpoint(model, optimizer, dataloader, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.config(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['step']
    val_loss = checkpoint['val_loss']
    print(f"Checkpoint loaded from step {step} with val loss {val_loss}")
    return step

def load_model(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    return
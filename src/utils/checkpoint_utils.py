import os
import torch

def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Saves the model weights, optimizer state, and current epoch to disk.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"\n[SAVE] Checkpoint successfully saved at epoch {epoch + 1} to {filepath}")

def load_checkpoint(model, optimizer, filepath, device):
    """
    Loads the training state if a checkpoint exists.
    Returns the epoch to resume from.
    """
    if os.path.isfile(filepath):
        print(f"\n[LOAD] Found checkpoint at {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
        
        print(f"[LOAD] Successfully restored model and optimizer. Resuming from epoch {start_epoch}\n")
        return start_epoch
    else:
        print("\n[START] No checkpoint found. Starting training from scratch.\n")
        return 0
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class TurtleBotVisionDataset(Dataset):
    def __init__(self, h5_path, models_dir, load_to_ram=True):
        print("Loading TurtleBot dataset into RAM...")
        self.f = h5py.File(h5_path, 'r')
        self.episodes = self.f['episode'][:]
        
        self.valid_indices = [
            i for i in range(len(self.episodes) - 1) 
            if self.episodes[i] == self.episodes[i+1] ]
        
        self.load_to_ram = load_to_ram
        if load_to_ram:
            self.rgb_data = self.f['rgb'][:]
            self.depth_data = self.f['depth'][:]
            self.state_data = self.f['state'][:]
            self.dist_data = self.f['distance'][:]
        
        # Normalization (5D state for TurtleBot)
        state_all = self.state_data if load_to_ram else self.f['state'][:]
        self.x_mean = np.mean(state_all, axis=0)
        self.x_std = np.std(state_all, axis=0)
        self.x_std[self.x_std < 1e-5] = 1.0
        
        os.makedirs(models_dir, exist_ok=True)
        np.save(Path(models_dir) / "tb3_x_mean.npy", self.x_mean)
        np.save(Path(models_dir) / "tb3_x_std.npy", self.x_std)
        print("Dataset fully loaded!")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        
        if self.load_to_ram:
            rgb = self.rgb_data[i].astype(np.float32) / 255.0
            dep = self.depth_data[i].astype(np.float32)
            state = self.state_data[i]
            d = self.dist_data[i][0]
            
            rgb_n = self.rgb_data[i+1].astype(np.float32) / 255.0
            dep_n = self.depth_data[i+1].astype(np.float32)
            state_n = self.state_data[i+1]
        else:
            # Fallback for disk reading omitted for brevity
            pass
            
        img = np.concatenate([rgb, dep], axis=0) 
        x = (state - self.x_mean) / self.x_std
        
        img_next = np.concatenate([rgb_n, dep_n], axis=0)
        x_next = (state_n - self.x_mean) / self.x_std
        
        return img, x, img_next, x_next, d
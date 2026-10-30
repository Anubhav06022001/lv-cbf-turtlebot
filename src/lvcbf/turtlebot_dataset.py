# import os
# import h5py
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path

# class TurtleBotVisionDataset(Dataset):
#     def __init__(self, h5_path, models_dir, load_to_ram=True):
#         print("Loading TurtleBot dataset into RAM...")
#         self.f = h5py.File(h5_path, 'r')
#         self.episodes = self.f['episode'][:]
        
#         self.valid_indices = [
#             i for i in range(len(self.episodes) - 1) 
#             if self.episodes[i] == self.episodes[i+1] ]
        
#         self.load_to_ram = load_to_ram
#         if load_to_ram:
#             self.rgb_data = self.f['rgb'][:]
#             self.depth_data = self.f['depth'][:]
#             self.state_data = self.f['state'][:]
#             self.dist_data = self.f['distance'][:]
        
#         # Normalization (5D state for TurtleBot)
#         state_all = self.state_data if load_to_ram else self.f['state'][:]
#         self.x_mean = np.mean(state_all, axis=0)
#         self.x_std = np.std(state_all, axis=0)
#         self.x_std[self.x_std < 1e-5] = 1.0
        
#         os.makedirs(models_dir, exist_ok=True)
#         np.save(Path(models_dir) / "tb3_x_mean.npy", self.x_mean)
#         np.save(Path(models_dir) / "tb3_x_std.npy", self.x_std)
#         print("Dataset fully loaded!")

#     def __len__(self):
#         return len(self.valid_indices)

#     def __getitem__(self, idx):
#         i = self.valid_indices[idx]
        
#         if self.load_to_ram:
#             rgb = self.rgb_data[i].astype(np.float32) / 255.0
#             dep = self.depth_data[i].astype(np.float32)
#             state = self.state_data[i]
#             d = self.dist_data[i][0]
            
#             rgb_n = self.rgb_data[i+1].astype(np.float32) / 255.0
#             dep_n = self.depth_data[i+1].astype(np.float32)
#             state_n = self.state_data[i+1]
#         else:
#             # Fallback for disk reading omitted for brevity
#             pass
            
#         img = np.concatenate([rgb, dep], axis=0) 
#         x = (state - self.x_mean) / self.x_std
        
#         img_next = np.concatenate([rgb_n, dep_n], axis=0)
#         x_next = (state_n - self.x_mean) / self.x_std
        
#         return img, x, img_next, x_next, d

# train2 chalane k lie
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class TurtleBotVisionDataset(Dataset):
    def __init__(self, h5_path, models_dir, load_to_ram=True, max_transitions=200_000, seed=0):
        print("Loading TurtleBot dataset (RGB only, subsampled)...")
        f = h5py.File(h5_path, 'r')

        episodes = f['episode'][:]
        state_all = f['state'][:].astype(np.float32)
        dist_all = f['distance'][:, 0].astype(np.float32)
        n_total = len(episodes)

        # Normalization from ALL states (cheap), saved next to the model
        self.x_mean = state_all.mean(axis=0)
        self.x_std = state_all.std(axis=0)
        self.x_std[self.x_std < 1e-5] = 1.0
        os.makedirs(models_dir, exist_ok=True)
        np.save(Path(models_dir) / "tb3_x_mean.npy", self.x_mean)
        np.save(Path(models_dir) / "tb3_x_std.npy", self.x_std)

        # Valid transitions: i -> i+1 within the same episode
        valid = np.where(episodes[:-1] == episodes[1:])[0]
        print(f"Total frames: {n_total} | valid transitions: {len(valid)}")
        if max_transitions and len(valid) > max_transitions:
            rng = np.random.default_rng(seed)
            valid = np.sort(rng.choice(valid, max_transitions, replace=False))
        print(f"Using {len(valid)} transitions")

        # Frames needed: i and i+1 for every chosen transition
        need = np.unique(np.concatenate([valid, valid + 1]))
        rgb = np.empty((len(need), 3, 64, 64), dtype=np.uint8)

        # Stream the file in contiguous blocks (fast, low peak memory)
        B = 20000
        for s in range(0, n_total, B):
            lo = np.searchsorted(need, s)
            hi = np.searchsorted(need, s + B)
            if hi > lo:
                blk = f['rgb'][s:s + B]
                rgb[lo:hi] = blk[need[lo:hi] - s]
        f.close()

        self.rgb = rgb
        self.pos_i = np.searchsorted(need, valid)
        self.pos_n = np.searchsorted(need, valid + 1)
        self.state_i = state_all[valid]
        self.state_n = state_all[valid + 1]
        self.dist = dist_all[valid]
        print(f"Dataset ready. RGB in RAM: {rgb.nbytes / 1e9:.2f} GB")

    def __len__(self):
        return len(self.dist)

    def __getitem__(self, idx):
        img = self.rgb[self.pos_i[idx]].astype(np.float32) / 255.0
        img_next = self.rgb[self.pos_n[idx]].astype(np.float32) / 255.0
        x = (self.state_i[idx] - self.x_mean) / self.x_std
        x_next = (self.state_n[idx] - self.x_mean) / self.x_std
        return img, x, img_next, x_next, self.dist[idx]
import torch
import torch.nn as nn
import torch.nn.functional as F 

class VisionBarrierNet(nn.Module):
    def __init__(self, state_dim=5, hidden_dim=256):
        super(VisionBarrierNet, self).__init__()

        # 1) ======== load DINO , freeze weights ========
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        for param in self.dino.parameters():
            param.requires_grad = False

        dino_out_dim = 384       # o/p dim of dino small 

        # 2) ======== NN for kinematic state =======
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU()
        )

        # 3) ========= fuse dino + kinematic state ==========
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dino_out_dim+64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

        # std imagenet normalisation for dino
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))


    def forward(self, img_4d, state):
        rgb = img_4d[: , :3,:, : ]      # extract first 3 channels rgb
        rgb_resized = F.interpolate(rgb, size=(224,224), mode='bilinear', align_corners=False)   # dino required image size to be multiple of 14
        rgb_norm = (rgb_resized - self.mean)/ self.std 

        with torch.no_grad():
            vis_features = self.dino(rgb_norm)

        state_features = self.state_mlp(state)

        combined = torch.cat([vis_features, state_features], dim=1)
        B_val = self.fusion_mlp(combined)

        return B_val




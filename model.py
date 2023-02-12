import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from primary import *
from encoder import *

class ProGRIP(nn.Module):
    def __init__(self,config):
        super().__init__()

        self.num_part = config.num_part
        self.num_pose = config.num_pose

        self.global_encoder = PointNetDenseEncoder(input_dim = config.input_dim)
        self.shape_render = FCBlock(config.latent_dim,3, 3 + config.latent_dim,1)

        # [Double Nested Transformer]
        self.shape_decoder = transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12,d_model = config.global_feature_dim,batch_first = True)
        self.pose_decoder = transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12,d_model = config.global_feature_dim,batch_first = True)

        key_dim = 128
        # [Decoder Keys for Double Nested Transformer]
        self.part_keys = nn.Parameter(torch.randn([self.num_part,key_dim]))
        self.pose_keys = nn.Parameter(torch.randn([self.num_pose,key_dim]))

    def point_transform(self,x,t,R,s):
        """
        x: [Nx3] # input volumetric point
        t: [Nx3] # translation for each point
        R: [Nx3x3] # SO(3) rotation for each point
        s: [Nx3] # input scale of each geometric shape on each aligned axis
        """
        return

    def forward(self,x):
        """
        inputs:
            x: point cloud datasets
        outputs:
            regular program shape
        """
        x = x # [B,200,3]
        # [Global Encoder]
        global_feature = self.global_encoder(x) # [B,128]
        
        # [Shape Decoder]
        axis_scale,part_latent = self.shape_decoder(global_feature,self.part_keys)
        # axis_scale: [B,N,3] shape_latent: [B,N,Z]

        # [Pose Decoder]
        

        return 0


if __name__ == "__main__":
    from config import *
    net = ProGRIP(config)
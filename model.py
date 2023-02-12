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
        shape_dim = config.shape_dim
        self.num_part = config.num_part
        self.num_pose = config.num_pose
        self.global_feature_dim = config.global_feature_dim

        self.global_encoder = PointNetDenseEncoder(input_dim = config.input_dim)
        self.shape_render = FCBlock(config.latent_dim,3, 3 + config.latent_dim,1)

        # [Double Nested Transformer]
        self.shape_decoder = nn.Transformer(nhead=16, num_encoder_layers=12,d_model = config.global_feature_dim,batch_first = True)
        self.pose_decoder = nn.Transformer(nhead=16, num_encoder_layers=12,d_model = config.global_feature_dim,batch_first = True)

        # [Shape Pose Feature Decoder]
        self.scale_para = FCBlock(128,3,self.global_feature_dim,3) # output the scale of the shape
        self.shape_para = FCBlock(128,3,self.global_feature_dim,shape_dim)# output the feature dim of decoder
        self.pose_para = FCTBlock(128,3,self.global_feature_dim,9)
        self.exist_para = FCBlock(128,3,self.global_feature_dim,1)

        key_dim = config.global_feature_dim
        # [Decoder Keys for Double Nested Transformer]
        self.part_keys = nn.Parameter(torch.randn([1,self.num_part,key_dim]))
        self.pose_keys = nn.Parameter(torch.randn([1,self.num_pose,key_dim]))
        
        # [Volume Shape Decoder]
        self.shape_decoder = FCBlock(128,3,self.global_feature_dim + 3,1) # decode the occupancy at point x

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
        B,n,_ = x.shape
        Z = self.global_feature_dim;M = self.num_pose;N = self.num_part
        x = x # [B,200,3]
        # [Global Encoder]
        global_feature = self.global_encoder(x) # [B,128]

        # [Part Decoder]
        input_part_queries = self.part_keys.repeat([B,1,1])
        part_feature= self.shape_decoder(global_feature.unsqueeze(1),input_part_queries)

        # [Pose Decoder]
        flat_source_seq = part_feature.reshape(-1,Z)
        input_pose_queries = self.pose_keys.unsqueeze(1).repeat([B,N,1,1]).reshape(-1,Z)
        pose_feature = self.shape_decoder(flat_source_seq,input_pose_queries).reshape([B,N,M,Z])

        # [Decoder Parameters]
        scales = self.scale_para(part_feature)
        features = self.shape_para(part_feature)

        pose_paras = self.pose_para(pose_feature)
        exist_paras = self.exist_para(pose_feature)
        print(scales.shape,features.shape)

        print(pose_paras.shape,exist_paras.shape)
        return 0


if __name__ == "__main__":
    from config import *
    net = ProGRIP(config)
    inputs = torch.randn([2,3,200])
    outputs = net(inputs)
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from primary import *
from encoder import *

from cuboid.network import Network_Whole

from utils import *

import json
from itertools import permutations

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
        self.trans_para = FCTBlock(128,3,self.global_feature_dim,3)
        self.exist_para = FCBlock(128,3,self.global_feature_dim,1)

        key_dim = config.global_feature_dim
        # [Decoder Keys for Double Nested Transformer]
        self.part_keys = nn.Parameter(torch.randn([1,self.num_part,key_dim]))
        self.pose_keys = nn.Parameter(torch.randn([1,self.num_pose,key_dim]))
        
        # [Volume Shape Decoder]
        self.geometric_decoder = FCBlock(128,3,self.global_feature_dim + 3,1) # decode the occupancy at point x

        # [Supervision Box Decoder] (use pretrain model)
        pretrain_path = config.state_root.format(config.category)
        hyppara_path = pretrain_path+'/hypara.json'
        state_path = pretrain_path + '/{}.pth'.format(config.category)
        with open(hyppara_path) as f:hyppara = json.load(f)
        self.supervision_box_decoder = Network_Whole(hyppara)
        self.supervision_box_decoder.load_state_dict(torch.load(state_path,map_location = config.device))
        self.config = config

    def point_transform(self,x,t,R,s):
        """
        x: [Nx3] # input volumetric point
        t: [Nx3] # translation for each point
        R: [Nx3x3] # SO(3) rotation for each point
        s: [Nx3] # input scale of each geometric shape on each aligned axis
        """
        return

    def forward(self,x,mode = "train_match"):
        """
        inputs:
            x: point cloud datasets [B,3,n-points]
        outputs:
            regular program shape
        """
        x = x.permute([0,2,1]) # [B,3,200]
        B,_,n = x.shape
        Z = self.global_feature_dim;M = self.num_pose;N = self.num_part

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
        scales = self.scale_para(part_feature).repeat(1,M,1)
        features = self.shape_para(part_feature)

        rotate_paras = self.pose_para(pose_feature).reshape([B,-1,3,3])
        trans_paras = self.trans_para(pose_feature).reshape([B,-1,3])
        exist_paras = self.exist_para(pose_feature).reshape([B,-1,1])


        if mode == "execute":
            hard_exist = (exist_paras + 0.5).int() # hard existence probability 
            return 1
        if mode == "train_match":
            config = self.config
            # calculate the matching loss of an object.

            # 1.[Calculate the Supervision Box]
            ground_box = self.supervision_box_decoder(x.permute([0,2,1]))
            gt_scale = ground_box["scale"];gt_rotate = ground_box["rotate"]
            gt_shift = ground_box["trans"];gt_exist = ground_box["exist"]
            # above section calculates the pseudo ground truth supervision
            
            gt_exist = torch.sigmoid(gt_exist)
            gt_hard_exist = (gt_exist + 0.5).int()
            pred_hard_exist = (exist_paras + 0.5).int()
            # 2.[Find Best Permutation] (Hugarian Match)
            batch_match_loss = 0
            # by the way, it does not support the batchwise operation
            for b in range(B): # enumerate over all batch
                # calculate the batch loss for using the Hungarian
                n_gt = gt_hard_exist.shape[1]
                n_pred = pred_hard_exist.shape[1]
                n_expand = max(n_gt,n_pred);n_contract = min(n_gt,n_pred)
                # create the cost matrix
                cost = torch.zeros([n_expand,n_expand])
                for i in range(n_gt):
                    for j in range(n_pred):
                        effective = gt_hard_exist[b][i] and pred_hard_exist[b][j]
                        gt_box_decode = decode_3d_box(gt_scale[b][i],gt_rotate[b][i],gt_shift[b][i])
                        pred_box_decode = decode_3d_box(scales[b][j],rotate_paras[b][j],trans_paras[b][j])

                        # box construction loss
                        if effective:
                            box_loss = 1 - box3d_iou(gt_box_decode,pred_box_decode)[0]
                        else: box_loss = 0.0                    
                        # existence loss

                        exist_loss = torch.nn.functional.binary_cross_entropy(exist_paras[b][j:j+1],gt_exist[b][i:i+1])

                        #print(box_loss,exist_loss)
                        pair_match_loss = config.l_s * box_loss * 0 + config.l_v * 0 + config.l_e * exist_loss
                        cost[i][j] += pair_match_loss
                try:
                    row_ind,col_ind = linear_sum_assignment(cost.detach())
                except:print(cost)
                batch_match_loss += cost[row_ind,col_ind].sum()
            # find the best permutation for the current predicted set.

            return {"match_loss":batch_match_loss}

        if mode == "train_execute":
            return 2
        return -1
    
    def calculate_matching_loss(self,input_pairs,target_pairs):
        return 0

if __name__ == "__main__":
    from config import *
    net = ProGRIP(config)
    inputs = torch.randn([2,200,3])
    outputs = net(inputs)
    print(outputs)

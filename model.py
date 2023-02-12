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
        self.shape_render = 

    def point_transform(self,x,t,R,s):
        """
        x: [Nx3] # input volumetric point
        t: [Nx3] # translation for each point
        R: [Nx3x3] # SO(3) rotation for each point
        s: [Nx1] # input scale of each geometric shape
        """
        return

    def forward(self,x):
        """
        inputs:
            x: point cloud datasets
        outputs:
            regular program shape
        """

        return 0
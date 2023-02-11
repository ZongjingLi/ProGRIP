import torch
import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt


class ProGRIP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_part = config.num_part
        self.num_pose = config.num_pose
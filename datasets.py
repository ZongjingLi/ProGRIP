import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

class ShapeNetDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self,idx):return idx

    def __len__(self):return 1
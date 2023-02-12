import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import os, sys, h5py
import numpy as np
import torch
import torch.utils.data as data


class ShapeNetDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self,idx):return idx

    def __len__(self):return 1


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class shapenet4096(data.Dataset):
    def __init__(self, phase, data_type, if_4096):
        super().__init__()
        data_root = "/Users/melkor/Documents/datasets/ShapeNetNormal4096/"
        self.folder = data_type + '/'
        if phase == 'train':
            self.data_list_file = data_root + data_type + '_train.npy'
        else:
            self.data_list_file = data_root + data_type + '_test.npy'
        self.data_dir = data_root + self.folder
        self.data_list = np.load(self.data_list_file)
        
    def __getitem__(self, idx):
        cur_name = self.data_list[idx].split('.')[0]
        cur_data = torch.from_numpy(np.load(self.data_dir + self.data_list[idx])).float()
        cur_points = cur_data[:,0:3]
        cur_normals = cur_data[:,3:]
        cur_points_num = 4096
        cur_values = -1
        return {"point_cloud":cur_points,
                "point_normal": cur_normals,
                "point_num": cur_points_num,
                "cur_values": cur_values, 
                "cur_names":cur_name}
        
    def __len__(self):
        return self.data_list.shape[0]

if __name__ == "__main__":

    dataset = shapenet4096("train","table",False)
    print(len(dataset))
    sample = dataset[1]
    cur_points = sample["point_cloud"]
    print(cur_points.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = cur_points[:,0]
    y = cur_points[:,1]
    z = cur_points[:,2]

    ax.scatter(x,y,z)
    plt.show()

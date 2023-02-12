import torch

device = "cuad:0" if torch.cuda.is_available() else "cpu"

import argparse

parser = argparse.ArgumentParser()

# add the input and network configuration
parser.add_argument("--input_dim",default = 3,help = "currently does not support 2d inputs.")
parser.add_argument("--latent_dim",default = 128,help = "the latent dim used for MLP")
parser.add_argument("--key_dim",default = 64,help = "the input dim to decode each shape and pose")
parser.add_argument("--frustum_size",default = (128,128,128), help = "size of the volumetric render")
parser.add_argument("--num_part",default = 4,help="number of differen parts being decoded by the geometric encoder.")
parser.add_argument("--num_pose",default = 5,help = "the number of pose a singe part decoder can be.")
parser.add_argument("--global_feature_dim",default = 256,help="the size of the global geometric encoder feature dim")
parser.add_argument("--shape_dim",default = 128,help = "shape of the P(x|z)")

# add the training configuration
parser.add_argument("--epoch",default = 100)
parser.add_argument("--batch_size",default = 4)
parser.add_argument("--shuffle",default = True)
parser.add_argument("--lr",default = 2e-4)
parser.add_argument("--dataset_name",default = "ShapeNet")
config = parser.parse_args(args = [])
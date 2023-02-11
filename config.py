import torch

device = "cuad:0" if torch.cuda.is_available() else "cpu"

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--latent_dim",default = 128)
parser.add_argument("--num_part",default = 4,help="number of differen parts being decoded by the geometric encoder.")
parser.add_argument("--num_pose",default = 4,help = "the number of pose a singe part decoder can be.")
parser.add_argument("--global_feature_dim",default = 128,help="the size of the global geometric encoder feature dim")
config = parser.parse_args(args = [])
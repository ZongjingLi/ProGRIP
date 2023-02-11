import torch

device = "cuad:0" if torch.cuda.is_available() else "cpu"

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--latent_dim",default = 128)
parser.add_argument("--num_parts",default = 4,help="number of differen parts being decoded by the geoemtric encoder")

config = parser.parse_args(args = [])
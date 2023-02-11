from datasets import *
from model import *

def train(model,dataset,config):
    print("start the ProGRIP training on the {}.".format(config.dataset_name))

if __name__ == "__main__":
    from config import *
    model = ProGRIP(config)
    dataset = ShapeNetDataset()
    train(model,dataset,config)
from datasets import *
from model import *

from tqdm import tqdm

def train(model,config):
    print("Start the ProGRIP training on the {}.".format(config.dataset_name))
    if config.dataset_name == "ShapeNet":
        dataset = ShapeNetDataset()
    dataloader = DataLoader(dataset,batch_size = config.batch_size,shuffle = config.shuffle)
    for epoch in range(1,config.epoch + 1):
        epoch_loss = 0
        for sample in tqdm(dataloader):
            sample
        
        print("epoch: {} loss: {}".format(epoch,epoch_loss))

if __name__ == "__main__":
    from config import *
    model = ProGRIP(config)
    train(model,config)
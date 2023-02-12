from datasets import *
from model import *

from tqdm import tqdm

def train(model,config):
    print("Start the ProGRIP training on the {}.".format(config.dataset_name))
    if config.dataset_name == "ShapeNet":
        dataset = ShapeNetDataset()
    dataloader = DataLoader(dataset,batch_size = config.batch_size,shuffle = config.shuffle)

    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    for epoch in range(1,config.epoch + 1):
        epoch_loss = 0
        for sample in tqdm(dataloader):
            sample
            # the input size should be a regular point cloud dataset. [BxNx3]

            # return size of the model should be... a volumetric render
            total_loss = 0.1

            # add the batch total loss to the epoch loss
            epoch_loss += total_loss

        
        print("epoch: {} loss: {}".format(epoch,epoch_loss))

if __name__ == "__main__":
    from config import *
    model = ProGRIP(config)
    train(model,config)
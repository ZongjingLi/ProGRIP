from datasets import *
from model import *

from tqdm import tqdm

def train_match(model,config):
    print("Start the ProGRIP training on the {}.".format(config.dataset_name))
    if config.dataset_name == "ShapeNet":
        dataset = shapenet4096("train",config.category,False)
    dataloader = DataLoader(dataset,batch_size = config.batch_size,shuffle = config.shuffle)

    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    for epoch in range(1,config.epoch + 1):
        epoch_loss = 0
        for sample in tqdm(dataloader):
            optimizer.zero_grad()
            
            point_cloud = sample["point_cloud"]
            outputs = model(point_cloud,"train_match")

            # the input size should be a regular point cloud dataset. [BxNx3]

            # return size of the model should be... a volumetric render
            total_loss = outputs["match_loss"]

            # backward optimize the model parameters
            total_loss.backward()
            optimizer.step()

            # add the batch total loss to the epoch loss
            epoch_loss += total_loss
        
        print("epoch: {} loss: {}".format(epoch,epoch_loss))


def train_execute(model,config):
    print("Start the ProGRIP execution training on {} dataset",format(config.dataset_name))
    if (config.datset_name == "ShapeNet"):
        dataset = shapenet4096("train",config.category,False)
    dataloader = DataLoader(dataset,batch_size = config.batch_size,shuffle = config.shuffle)

    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    for epoch in range(1,config.epoch + 1):
        epoch_loss = 0
        for sample in tqdm(dataloader):

            optimizer.zero_grad()
            
            point_cloud = sample["point_cloud"]
            outputs = model(point_cloud,"train_execute")

            # the input size should be a regular point cloud data with fixed point number
            total_loss = outputs["execute_loss"]

            # backward optimmize the model parameters
            total_loss.backward()
            optimizer.step()

            # add the batch total loss to the epoch loss
            epoch_loss += total_loss
        print("epoch: {} loss: {}".format(epoch,epoch_loss))

if __name__ == "__main__":
    from config import *
    model = ProGRIP(config)
    train_match(model,config)
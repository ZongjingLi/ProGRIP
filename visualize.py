from cuboid.network import Network_Whole
from utils import *
import json

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from datasets import *
from config import *
cat = "animal"
dataset = dataset = shapenet4096("train",cat,False)


root = "/Users/melkor/Documents/GitHub/ProGRIP/cuboid/PretrainModels/{}".format(cat)
with open(root+ '/hypara.json') as f:
    hypara = json.load(f) 
net = Network_Whole(hypara)
net.load_state_dict(torch.load(root + "/{}.pth".format(cat),map_location = device))

pc = dataset[0]["point_cloud"].unsqueeze(0)

outputs = net(pc)

scale = outputs["scale"]
rot = outputs["rotate"]
shift = outputs["trans"]
exist = outputs["exist"]

# point cloud cuboid summary created
b = 0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = [-1,1]
X, Y = np.meshgrid(r, r)
for k in range(14):
    boxx = decode_3d_box(scale[b][k],rot[b][k],shift[b][k])
    points = np.array(boxx.detach())
    Z = points * 1
    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2],color="cyan")
    verts = [[Z[0],Z[1],Z[2],Z[3]],
    [Z[4],Z[5],Z[6],Z[7]],
    [Z[0],Z[1],Z[5],Z[4]],
    [Z[2],Z[3],Z[7],Z[6]],
    [Z[1],Z[2],Z[6],Z[5]],
    [Z[4],Z[7],Z[3],Z[0]]]
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))

ax.scatter3D(pc[b,:,0],pc[b,:,1],pc[b,:,2])

points = torch.tensor([[-1, -1, -1],
                  [1, -1, -1 ],
                  [1, 1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1],
                  [1, -1, 1 ],
                  [1, 1, 1],
                  [-1, 1, 1]]) * 0.5

ax.scatter3D(points[:,1],points[:,2],points[:,0])

plt.show()
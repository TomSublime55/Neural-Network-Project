import torch
import torch.nn as nn
import numpy as np

file = np.load('DataTrain.npy')
data = torch.tensor(file)
print(data)
print(data.size())
size = data.size()

loader = torch.utils.data.DataLoader(data, batch_size=100000, shuffle=True)

class Model(nn.Module):
  def __init__(self, layerSize = [128, 64, 32, 16]):
    super(Model, self).__init__()
    layers = []
    layers.append(nn.Flatten())
    c = size[1]
    for s in range(len(layerSize) - 1):
      layers.append(nn.Linear(c, s))
      layers.append(nn.ReLU())
      c = s
    layers.append(nn.Linear(c, 1))
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)

model = Model()

loss = nn.BCEWithLogitsLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.001)

import torch.nn as nn
import torch
import torch.nn.functional as F

dimension = 256

class DecoderSDF(nn.Module):
    def __init__(self, latent_size):
        super(DecoderSDF, self).__init__()

        self.linearStart = nn.Linear(latent_size + 3, dimension)

        self.linear1= nn.Linear(dimension, dimension)
        self.linear2= nn.Linear(dimension, dimension)
        self.linear3 = nn.Linear(dimension, dimension)
        self.linear4 = nn.Linear(dimension, dimension)
        self.linear5 = nn.Linear(dimension, dimension)

        self.linearEnd = nn.Linear(dimension, 4)

        self.relu = nn.ReLU()

        self.sgm = nn.Sigmoid()

        self.bn0 = nn.BatchNorm1d(dimension)
        self.bn1 = nn.BatchNorm1d(dimension)
        self.bn2 = nn.BatchNorm1d(dimension)
        self.bn3 = nn.BatchNorm1d(dimension)
        self.bn4 = nn.BatchNorm1d(dimension)
        self.bn5 = nn.BatchNorm1d(dimension)
    
    def forward(self, latent_code, xyz):
        
        x = torch.cat([latent_code, xyz], dim=1)

        x = self.linearStart(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.linear4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.linear5(x)
        x = self.bn5(x)
        x = self.relu(x)


        x = self.linearEnd(x)

        x[:,1:] = self.sgm(3 * x[:,1:])

        return x 

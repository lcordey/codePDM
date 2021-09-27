
import torch.nn as nn
import torch
import torch.nn.functional as F

dimension = 256

class DecoderSDF(nn.Module):
    def __init__(self, latent_size):
        super(DecoderSDF, self).__init__()

        self.linearStart = nn.Linear(latent_size + 3, dimension)

        self.linearStartRgb = nn.Linear(latent_size + 3 + 1, dimension)

        self.linear1= nn.Linear(dimension, dimension)
        self.linear2= nn.Linear(dimension, dimension)
        self.linear3 = nn.Linear(dimension, dimension)
        self.linear4 = nn.Linear(dimension, dimension)
        self.linear5 = nn.Linear(dimension, dimension)

        self.linear6= nn.Linear(dimension, dimension)
        self.linear7= nn.Linear(dimension, dimension)
        self.linear8 = nn.Linear(dimension, dimension)
        self.linear9 = nn.Linear(dimension, dimension)
        self.linear10 = nn.Linear(dimension, dimension)

        self.linearEnd = nn.Linear(dimension, 4)

        self.linearEndSdf = nn.Linear(dimension, 1)
        self.linearEndRgb = nn.Linear(dimension, 3)

        self.relu = nn.ReLU()

        self.sgm = nn.Sigmoid()

        self.bn0 = nn.BatchNorm1d(dimension)
        self.bn1 = nn.BatchNorm1d(dimension)
        self.bn2 = nn.BatchNorm1d(dimension)
        self.bn3 = nn.BatchNorm1d(dimension)
        self.bn4 = nn.BatchNorm1d(dimension)
        self.bn5 = nn.BatchNorm1d(dimension)
        self.bn6 = nn.BatchNorm1d(dimension)
        self.bn7 = nn.BatchNorm1d(dimension)
    
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

        # x = self.linear6(x)
        # x = self.bn6(x)
        # x = self.relu(x)

        # x = self.linear7(x)
        # x = self.bn7(x)
        # x = self.relu(x)

        # x = self.linear8(x)
        # x = self.relu(x)

        # x = self.linear9(x)
        # x = self.relu(x)

        # x = self.linear10(x)
        # x = self.relu(x)

        x = self.linearEnd(x)

        x[:,1:] = self.sgm(3 * x[:,1:])

        return x 


        # input_sdf = torch.cat([latent_code, xyz], dim=1)

        # sdf = self.linearStart(input_sdf)
        # sdf = self.relu(sdf)

        # sdf = self.linear1(sdf)
        # sdf = self.bn(sdf)

        # sdf = self.linear2(sdf)
        # sdf = self.bn(sdf)

        # sdf = self.linear3(sdf)
        # sdf = self.bn(sdf)

        # sdf = self.linear4(sdf)
        # sdf = self.bn(sdf)

        # sdf = self.linear5(sdf)
        # sdf = self.bn(sdf)

        # sdf = self.linearEndSdf(sdf)

        # input_rgb = torch.cat([input_sdf,sdf],dim=1)

        
        # rgb = self.linearStartRgb(input_rgb)
        # rgb = self.relu(rgb)


        # rgb = self.linear6(rgb)
        # rgb = self.bn(rgb)

        # rgb = self.linear7(rgb)
        # rgb = self.bn(rgb)

        # rgb = self.linear8(rgb)
        # rgb = self.bn(rgb)

        # rgb = self.linear9(rgb)
        # rgb = self.bn(rgb)

        # rgb = self.linear10(rgb)
        # rgb = self.bn(rgb)

        # rgb = self.linearEndRgb(rgb)
        # rgb = self.sgm(rgb)

        # sdf_rgb = torch.cat([sdf,rgb],dim=1)


        # return sdf_rgb
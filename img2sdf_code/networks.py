import torch.nn as nn
import torch

import IPython

def fc_layer(size_in, size_out, batch_norm=False):

    layers = [nn.Linear(size_in, size_out)]
    if batch_norm:
        layers += [nn.BatchNorm1d(size_out)]
    layers += [nn.ReLU()]

    return nn.Sequential(*layers)


class Decoder(nn.Module):
    def __init__(self, latent_size, batch_norm=False):
        super(Decoder, self).__init__()

        # num_features = 256

        # self.lnStart = fc_layer(latent_size + 3, num_features, batch_norm=batch_norm)

        # self.ln1 = fc_layer(num_features, num_features, batch_norm=batch_norm)
        # self.ln2 = fc_layer(num_features, num_features, batch_norm=batch_norm)
        # self.ln3 = fc_layer(num_features, num_features, batch_norm=batch_norm)
        # self.ln4 = fc_layer(num_features, num_features, batch_norm=batch_norm)
        # self.ln5 = fc_layer(num_features, num_features, batch_norm=batch_norm)

        # self.lnEnd = nn.Linear(num_features, 4)

        # self.sgm = nn.Sigmoid()
        # self.lambda_activation = 3


        dimension = 256
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
        
        # x = torch.cat([latent_code, xyz], dim=1)

        # x = self.lnStart(x)
        # x = self.ln1(x)
        # x = self.ln2(x)
        # x = self.ln3(x)
        # x = self.ln4(x)
        # x = self.ln5(x)

        # x = self.lnEnd(x)

        # # activation function, only for rgb values
        # x[:,1:] = self.sgm(self.lambda_activation * x[:,1:])


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

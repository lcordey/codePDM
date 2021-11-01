import torch.nn as nn
import torch

import IPython

def fc_layer(size_in, size_out, batch_norm=False):

    layers = [nn.Linear(size_in, size_out)]
    # if batch_norm:
    #     layers += [nn.BatchNorm1d(size_out)]
    layers += [nn.ReLU()]

    return nn.Sequential(*layers)


class Decoder(nn.Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()

        num_features = 256

        self.lnStart = fc_layer(latent_size + 3, num_features, batch_norm=True)

        self.ln1 = fc_layer(num_features, num_features, batch_norm=True)
        self.ln2 = fc_layer(num_features, num_features, batch_norm=True)
        self.ln3 = fc_layer(num_features, num_features, batch_norm=True)
        self.ln4 = fc_layer(num_features, num_features, batch_norm=True)
        self.ln5 = fc_layer(num_features, num_features, batch_norm=True)

        self.lnEnd = nn.Linear(num_features, 4)

        self.sgm = nn.Sigmoid()
        self.lambda_activation = 3
    
    def forward(self, latent_code, xyz):
        
        x = torch.cat([latent_code, xyz], dim=1)

        x = self.lnStart(x)
        x = self.ln1(x)
        x = self.ln2(x)
        x = self.ln3(x)
        x = self.ln4(x)
        x = self.ln5(x)

        x = self.lnEnd(x)

        # activation function, only for rgb values
        x[:,1:] = self.sgm(self.lambda_activation * x[:,1:])

        return x 

import torch.nn as nn
import torch

import IPython


def conv_layer3D(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv3d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        # nn.BatchNorm3d(chann_out),
        nn.ReLU()
    )
    return layer

def conv_block3D(in_list, out_list, k_list, p_list, pooling_k):

    layers = [ conv_layer3D(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool3d(kernel_size = pooling_k)]
    return nn.Sequential(*layers)


def fc_layer(size_in, size_out, batch_norm=False):

    layers = [nn.Linear(size_in, size_out)]
    if batch_norm:
        layers += [nn.BatchNorm1d(size_out)]
    layers += [nn.ReLU()]

    return nn.Sequential(*layers)


class Decoder(nn.Module):
    def __init__(self, latent_size, batch_norm=False):
        super(Decoder, self).__init__()

        num_features = 256

        self.lnStart = fc_layer(latent_size + 3, num_features, batch_norm=batch_norm)

        self.ln1 = fc_layer(num_features, num_features, batch_norm=batch_norm)
        self.ln2 = fc_layer(num_features, num_features, batch_norm=batch_norm)
        self.ln3 = fc_layer(num_features, num_features, batch_norm=batch_norm)
        self.ln4 = fc_layer(num_features, num_features, batch_norm=batch_norm)
        self.ln5 = fc_layer(num_features, num_features, batch_norm=batch_norm)

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

class EncoderGrid(nn.Module):
    def __init__(self,latent_size):
        super(EncoderGrid, self).__init__()

        features_encoder = 64

        self.block1 = conv_block3D([3,features_encoder], [features_encoder,features_encoder], [3,3], [1,1], 2)
        self.block2 = conv_block3D([features_encoder,features_encoder], [features_encoder,features_encoder], [3,3], [1,1], 2)
        self.block3 = conv_block3D([features_encoder,2 * features_encoder], [2 * features_encoder, 2 * features_encoder], [3,3], [1,1], 2)

        self.fc1 = fc_layer(6*3*3*features_encoder * 2, features_encoder)
        self.fc2 = fc_layer(features_encoder, features_encoder)
        self.fc3 = fc_layer(features_encoder, features_encoder)
        self.fc4 = nn.Linear(features_encoder, latent_size)

    def forward(self, image):

        temp = self.block1(image)
        temp = self.block2(temp)
        features = self.block3(temp)

        latent_code = self.fc4(self.fc3(self.fc2(self.fc1(features.view(features.size(0), -1)))))

        return latent_code

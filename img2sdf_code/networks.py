from pickle import FALSE
import torch.nn as nn
import torch
import math

import IPython


def conv_layer3D(chann_in, chann_out, k_size, p_size, batch_norm=False):

    layers = [nn.Conv3d(chann_in, chann_out, kernel_size=k_size, padding=p_size)]
    if batch_norm:
        layers += [nn.BatchNorm3d(chann_out)]
    layers += [nn.ReLU()]

    return nn.Sequential(*layers)

def conv_block3D(in_list, out_list, k_list, p_list, pooling_k, batch_norm=FALSE):

    layers = [ conv_layer3D(in_list[i], out_list[i], k_list[i], p_list[i], batch_norm) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool3d(kernel_size = pooling_k)]

    return nn.Sequential(*layers)

def features_extraction_conv_block3D(num_block, num_features, k_list, p_list, pooling_k, batch_norm=FALSE):
    layers  = [conv_block3D([3, num_features], [num_features, num_features], k_list, p_list, pooling_k, batch_norm=batch_norm)]
    for i in range(1, num_block):
        layers += [conv_block3D([num_features, num_features], [num_features, num_features], k_list, p_list, pooling_k, batch_norm=batch_norm)]

    return nn.Sequential(*layers)

def fc_layer(chann_in, chann_out, batch_norm=False):

    layers = [nn.Linear(chann_in, chann_out)]
    if batch_norm:
        layers += [nn.BatchNorm1d(chann_out)]
    layers += [nn.ReLU()]

    return nn.Sequential(*layers)

def fc_block(num_fc_layer, num_features, num_features_extracted, latent_size, batch_norm=False):
    layers = [fc_layer(num_features_extracted * num_features, num_features, batch_norm=batch_norm)]
    for i in range(1, num_fc_layer):
        layers += [fc_layer(num_features, num_features, batch_norm=batch_norm)]

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
    def __init__(self,latent_size, param, vae=False,batch_norm_conv=False, batch_norm_fc=False):
        super(EncoderGrid, self).__init__()

        self.latent_size = latent_size
        self.vae = vae

        features_encoder = 64

        # block of multiple convolution layer
        self.features_extraction = features_extraction_conv_block3D(param["num_conv_layer"], features_encoder, [3,3], [1,1], 2, batch_norm=batch_norm_conv)

        # compute size of resulting feature space
        num_slice_features = math.floor(param["num_slices"] / 2**param["num_conv_layer"])
        num_width_features = math.floor(param["width"] / 2**param["num_conv_layer"])
        num_height_features = math.floor(param["height"] / 2**param["num_conv_layer"])
        num_features_extracted = num_slice_features * num_width_features * num_height_features

        assert(num_slice_features > 0 and num_width_features > 0 and num_height_features > 0), "too many conv layers"
        print(f"Init Encoder, features size of: {num_slice_features} x {num_width_features} x {num_height_features}")

        # MLP
        self.MLP = fc_block(param["num_fc_layer"], features_encoder, num_features_extracted, latent_size, batch_norm=batch_norm_fc)

        # last fc layer used for regression to sdf prediction
        if not vae:
            self.classifier= nn.Linear(features_encoder, latent_size)
        else:
            self.classifier= nn.Linear(features_encoder, 2 * latent_size)


    def forward(self, image):

        # extract features from conv layer
        features = self.features_extraction(image)

        # flatten features
        features = features.view(features.size(0), -1)

        # get code from features
        features = self.MLP(features)

        if not self.vae:
            latent_code = self.classifier(features)
            return latent_code
        else:
            features = self.classifier(features)
            latent_code_mu = features[:self.latent_size]
            latent_code_log_std = features[self.latent_size:]
            return latent_code_mu, latent_code_log_std


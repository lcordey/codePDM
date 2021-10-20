import torch.nn as nn
import torch
import torch.nn.functional as F

import time
import IPython


class DecoderSDF(nn.Module):
    def __init__(self, latent_size):
        super(DecoderSDF, self).__init__()

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


class EncoderSDF(nn.Module):
    def __init__(self,latent_size, vae = False):
        super(EncoderSDF, self).__init__()

        features_encoder = 32

        self.conv1 = nn.Conv2d(3, (int)(features_encoder/4), kernel_size=(3, 3))
        self.conv2 = nn.Conv2d((int)(features_encoder/4), (int)(features_encoder/2), kernel_size=(3, 3))

        self.conv3 = nn.Conv2d((int)(features_encoder/2), features_encoder, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3, 3))

        self.conv5 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3, 3))
        self.conv6 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3, 3))

        self.conv9 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3, 3))
        self.conv10 = nn.Conv2d(features_encoder, (int)(features_encoder/2), kernel_size=(3, 3))

        self.conv11 = nn.Conv2d((int)(features_encoder/2), (int)(features_encoder/4), kernel_size=(3, 3))
        self.conv12 = nn.Conv2d((int)(features_encoder/4), 1, kernel_size=(3, 3))

        self.maxpool1 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(5 * 10 + 20, 2*features_encoder)
        self.linear2 = nn.Linear(2*features_encoder, 4*features_encoder)
        self.linear3 = nn.Linear(4*features_encoder, 4*features_encoder)
        self.linear4 = nn.Linear(4*features_encoder, 2*features_encoder)
        
        if not vae:
            self.linear5 = nn.Linear(2*features_encoder, latent_size)
        else:
            self.linear5 = nn.Linear(2*features_encoder, 2 * latent_size)



        #  batch norm
        #  dropout
        #  dilation
        #  batch size


        self.relu = nn.ReLU()

     
    def forward(self, image, loc):
        # image: N x 3 x 300 x 450
        # loc N x 20

        image = self.conv1(image)
        image = self.conv2(image)
        image = self.relu(image)
        image = self.maxpool1(image)

        image = self.conv3(image)
        image = self.conv4(image)
        image = self.relu(image)
        image = self.maxpool1(image)

        image = self.conv5(image)
        image = self.conv6(image)
        image = self.relu(image)
        image = self.maxpool1(image)

        image = self.conv9(image)
        image = self.conv10(image)
        image = self.relu(image)
        image = self.maxpool1(image)

        image = self.conv11(image)
        image = self.conv12(image)
        image = self.relu(image)
        image = self.maxpool1(image)

        # print(image.shape)
        image = torch.flatten(image, start_dim=1)
        merged = torch.cat([image,loc], dim = 1)

        latent_code = self.linear5(self.relu(self.linear4(self.relu(self.linear3(self.relu(self.linear2(self.relu(self.linear1(merged)))))))))


        return latent_code


class EncoderGrid(nn.Module):
    def __init__(self,latent_size):
        super(EncoderGrid, self).__init__()

        features_encoder = 64

        self.conv1 = nn.Conv3d(3, features_encoder, kernel_size=(5,3,3))
        self.conv2 = nn.Conv3d(features_encoder, features_encoder, kernel_size=(5,3,3))
        self.conv3 = nn.Conv3d(features_encoder, features_encoder, kernel_size=(5,3,3))
        self.conv4 = nn.Conv3d(features_encoder, features_encoder, kernel_size=(5,3,3))

        self.mp = nn.MaxPool3d(2)

        self.ln1 = nn.Linear(6*3*3 * features_encoder, features_encoder)
        self.ln2 = nn.Linear(features_encoder, features_encoder)
        self.ln3 = nn.Linear(features_encoder, features_encoder)
        self.ln4 = nn.Linear(features_encoder, latent_size)

        self.relu = nn.ReLU()

    def forward(self, image):
        image = self.conv2(self.conv1(image))
        image = self.mp(image)
        
        image = self.conv4(self.conv3(image))
        image = self.mp(image)

        # print(image.shape)

        image = torch.flatten(image, start_dim=1)

        image = self.relu(self.ln1(image))
        image = self.relu(self.ln2(image))
        image = self.relu(self.ln3(image))

        latent_code = self.ln4(image)

        return latent_code

class EncoderFace(nn.Module):
    def __init__(self,latent_size):
        super(EncoderFace, self).__init__()

        features_encoder = 64

        # front
        self.convFront1 = nn.Conv2d(3, features_encoder, kernel_size=(3,3))
        self.convFront2 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3,3))
        self.convFront3 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3,3))
        self.convFront4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3,3))
        self.convFront4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3,3))
        self.convFront4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3,3))

        self.lnFront1 = nn.Linear(5*5 * features_encoder, features_encoder)
        self.lnFront1 = nn.Linear(features_encoder, features_encoder)
        self.lnFront3 = nn.Linear(features_encoder, latent_size)

        # left
        self.convLeft1 = nn.Conv2d(3, features_encoder, kernel_size=(5,3))
        self.convLeft2 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convLeft3 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convLeft4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convLeft4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convLeft4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))

        self.lnLeft1 = nn.Linear(8*5 * features_encoder, features_encoder)
        self.lnLeft1 = nn.Linear(features_encoder, features_encoder)
        self.lnLeft3 = nn.Linear(features_encoder, latent_size)

        # back
        self.convBack1 = nn.Conv2d(3, features_encoder, kernel_size=(3,3))
        self.convBack2 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3,3))
        self.convBack3 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3,3))
        self.convBack4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3,3))
        self.convBack4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3,3))
        self.convBack4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3,3))

        self.lnBack1 = nn.Linear(5*5 * features_encoder, features_encoder)
        self.lnBack1 = nn.Linear(features_encoder, features_encoder)
        self.lnBack3 = nn.Linear(features_encoder, latent_size)

        # right
        self.convRight1 = nn.Conv2d(3, features_encoder, kernel_size=(5,3))
        self.convRight2 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convRight3 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convRight4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convRight4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convRight4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))

        self.lnRight1 = nn.Linear(8*5 * features_encoder, features_encoder)
        self.lnRight1 = nn.Linear(features_encoder, features_encoder)
        self.lnRight3 = nn.Linear(features_encoder, latent_size)

        # top
        self.convTop1 = nn.Conv2d(3, features_encoder, kernel_size=(5,3))
        self.convTop2 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convTop3 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convTop4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convTop4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))
        self.convTop4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(5,3))

        self.lnTop1 = nn.Linear(8*5 * features_encoder, features_encoder)
        self.lnTop1 = nn.Linear(features_encoder, features_encoder)
        self.lnTop3 = nn.Linear(features_encoder, latent_size)


        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, image):
        image = self.conv2(self.conv1(image))
        image = self.mp(image)
        
        image = self.conv4(self.conv3(image))
        image = self.mp(image)

        # print(image.shape)

        image = torch.flatten(image, start_dim=1)

        image = self.relu(self.ln1(image))
        image = self.relu(self.ln2(image))
        image = self.relu(self.ln3(image))

        latent_code = self.ln4(image)

        return latent_code    

code = torch.empty(10,16)
xyz_coord = torch.empty(10,3)

decoder = DecoderSDF(16)

decoder(code,xyz_coord)
pytorch_total_params = sum(p.numel() for p in decoder.parameters())
print("Decoder paramter: {}".format(pytorch_total_params))


image = torch.empty(10,3,300,450)
loc = torch.empty(10,20)

encoder = EncoderSDF(16)

encoder(image,loc)
pytorch_total_params = sum(p.numel() for p in encoder.parameters())
print("Encoder paramter: {}".format(pytorch_total_params))



grid = torch.empty(10,3,50,25,25)

encoder = EncoderGrid(16)

encoder(grid)
pytorch_total_params = sum(p.numel() for p in encoder.parameters())
print("Encoder paramter: {}".format(pytorch_total_params))


# width, height, depth = 100, 50, 50

# front = torch.empty(10,3,width,height)
# left = torch.empty(10,3,depth,height)
# back = torch.empty(10,3,width,height)
# right = torch.empty(10,3,depth,height)
# top = torch.empty(10,3,width,depth)

# encoder = EncoderFace(16)

# encoder(front, left, back, right, top)
# pytorch_total_params = sum(p.numel() for p in encoder.parameters())
# print("Encoder paramter: {}".format(pytorch_total_params))
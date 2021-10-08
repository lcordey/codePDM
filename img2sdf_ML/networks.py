import torch.nn as nn
import torch
import torch.nn.functional as F

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
    def __init__(self,latent_size):
        super(EncoderSDF, self).__init__()

        features_encoder = 32

        self.conv1 = nn.Conv2d(3, (int)(features_encoder/4), kernel_size=(3, 3))
        self.conv2 = nn.Conv2d((int)(features_encoder/4), (int)(features_encoder/2), kernel_size=(3, 3))

        self.conv3 = nn.Conv2d((int)(features_encoder/2), features_encoder, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3, 3))

        self.conv5 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3, 3))
        self.conv6 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3, 3))

        # self.conv7 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3, 3))
        # self.conv8 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3, 3))

        self.conv9 = nn.Conv2d(features_encoder, features_encoder, kernel_size=(3, 3))
        self.conv10 = nn.Conv2d(features_encoder, (int)(features_encoder/2), kernel_size=(3, 3))

        self.conv11 = nn.Conv2d((int)(features_encoder/2), (int)(features_encoder/4), kernel_size=(3, 3))
        self.conv12 = nn.Conv2d((int)(features_encoder/4), 1, kernel_size=(3, 3))

        self.maxpool1 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(5 * 10 + 20, features_encoder)
        self.linear2 = nn.Linear(features_encoder, features_encoder)
        self.linear3 = nn.Linear(features_encoder, latent_size)
        # self.linear2 = nn.Linear(20, 2)




        self.relu = nn.ReLU()

     
    def forward(self, image, loc):
        # image: N x 3 x 300 x 450
        # loc N x 20

        # print(image.shape)
        # print(loc.shape)

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

        # image = self.conv7(image)
        # image = self.conv8(image)
        # image = self.relu(image)
        # image = self.maxpool1(image)

        image = self.conv9(image)
        image = self.conv10(image)
        image = self.relu(image)
        image = self.maxpool1(image)

        # print(image.shape)

        image = self.conv11(image)
        image = self.conv12(image)
        image = self.relu(image)
        image = self.maxpool1(image)

        # print(image.shape)
        image = torch.flatten(image, start_dim=1)


        merged = torch.cat([image,loc], dim = 1)

        # print(merged.shape)
        latent_code = self.linear3(self.relu(self.linear2(self.relu(self.linear1(merged)))))


        return latent_code



image = torch.empty(10,3,300,450)
loc = torch.empty(10,20)

model = EncoderSDF(16)

model(image,loc)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
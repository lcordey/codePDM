import torch.nn as nn
import torch
import torch.nn.functional as F

import time
import IPython

def conv_layer3D(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv3d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        # nn.BatchNorm3d(chann_out),
        # nn.ReLU()
    )
    return layer

def conv_block3D(in_list, out_list, k_list, p_list, pooling_k):

    layers = [ conv_layer3D(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool3d(kernel_size = pooling_k)]
    return nn.Sequential(*layers)


def conv_layer2D(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer

def conv_block2D(in_list, out_list, k_list, p_list, pooling_k):

    layers = [ conv_layer2D(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ nn.MaxPool2d(kernel_size = pooling_k)]
    return nn.Sequential(*layers)

def fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        # nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


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
        self.ln3 = nn.Linear(features_encoder, (int)(features_encoder/2))
        self.ln4 = nn.Linear((int)(features_encoder/2), latent_size)

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



class EncoderGrid2(nn.Module):
    def __init__(self,latent_size):
        super(EncoderGrid2, self).__init__()

        features_encoder = 64

        # self.block1 = conv_block3D([3,features_encoder], [features_encoder,features_encoder], [(3,3,3),(3,3,3)], [(1,1,1),(1,1,1)], 2)
        # self.block2 = conv_block3D([features_encoder,features_encoder], [features_encoder,features_encoder], [(3,3,3),(3,3,3)], [(1,1,1),(1,1,1)], 2)
        # self.block3 = conv_block3D([features_encoder,2 * features_encoder], [2 * features_encoder, 2 * features_encoder], [(3,3,3),(3,3,3)], [(1,1,1),(1,1,1)], 2)

        self.block1 = conv_block3D([3,features_encoder], [features_encoder,features_encoder], [(5,3,3),(5,3,3)], [0,0], 2)
        self.block3 = conv_block3D([features_encoder, features_encoder], [features_encoder, features_encoder], [(5,3,3),(5,3,3)], [0,0], 2)

        # self.fc1 = fc_layer(6*3*3*features_encoder * 2, features_encoder)
        self.fc1 = fc_layer(6*3*3*features_encoder, features_encoder)
        self.fc2 = fc_layer(features_encoder, features_encoder)
        self.fc3 = fc_layer(features_encoder, (int)(features_encoder/2))
        self.fc4 = fc_layer((int)(features_encoder/2), latent_size)

    def forward(self, image):

        temp = self.block1(image)
        # temp = self.block2(temp)
        features = self.block3(temp)

        temp = torch.flatten(features, start_dim=1)
        latent_code = self.fc4(self.fc3(self.fc2(self.fc1(temp))))

        return latent_code


class EncoderFace(nn.Module):
    def __init__(self,latent_size):
        super(EncoderFace, self).__init__()

        features_encoder = 64

        # front
        self.blockFront1 = conv_block2D([3,features_encoder], [features_encoder,features_encoder], [(3,3),(3,3)], [1,1], 2)
        self.blockFront2 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(3,3),(3,3)], [1,1], 2)
        self.blockFront3 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(3,3),(3,3)], [1,1], 2)
        self.blockFront4 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(3,3),(3,3)], [1,1], 2)

        self.blockLeft1 = conv_block2D([3,features_encoder], [features_encoder,features_encoder], [(5,3),(5,3)], [(2,1),(2,1)], 2)
        self.blockLeft2 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(5,3),(5,3)], [(2,1),(2,1)], 2)
        self.blockLeft3 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(5,3),(5,3)], [(2,1),(2,1)], 2)
        self.blockLeft4 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(5,3),(5,3)], [(2,1),(2,1)], 2)

        self.blockBack1 = conv_block2D([3,features_encoder], [features_encoder,features_encoder], [(3,3),(3,3)], [1,1], 2)
        self.blockBack2 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(3,3),(3,3)], [1,1], 2)
        self.blockBack3 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(3,3),(3,3)], [1,1], 2)
        self.blockBack4 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(3,3),(3,3)], [1,1], 2)

        self.blockRight1 = conv_block2D([3,features_encoder], [features_encoder,features_encoder], [(5,3),(5,3)], [(2,1),(2,1)], 2)
        self.blockRight2 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(5,3),(5,3)], [(2,1),(2,1)], 2)
        self.blockRight3 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(5,3),(5,3)], [(2,1),(2,1)], 2)
        self.blockRight4 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(5,3),(5,3)], [(2,1),(2,1)], 2)

        self.blockTop1 = conv_block2D([3,features_encoder], [features_encoder,features_encoder], [(3,5),(3,5)], [(1,2),(1,2)], 2)
        self.blockTop2 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(3,5),(3,5)], [(1,2),(1,2)], 2)
        self.blockTop3 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(3,5),(3,5)], [(1,2),(1,2)], 2)
        self.blockTop4 = conv_block2D([features_encoder, features_encoder], [features_encoder, features_encoder], [(3,5),(3,5)], [(1,2),(1,2)], 2)


        self.ln1 = fc_layer((2*4*4 + 3*4*8) * features_encoder, features_encoder)
        self.ln2 = fc_layer(features_encoder, features_encoder)
        self.ln3 = fc_layer(features_encoder, latent_size)

    def forward(self, front, left, back, right, top):

        latent_code = None

        features_front = self.blockFront4(self.blockFront3(self.blockFront2(self.blockFront1(front))))
        features_left = self.blockLeft4(self.blockLeft3(self.blockLeft2(self.blockLeft1(left))))
        features_back = self.blockBack4(self.blockBack3(self.blockBack2(self.blockBack1(back))))
        features_right = self.blockRight4(self.blockRight3(self.blockRight2(self.blockRight1(right))))
        features_top = self.blockTop4(self.blockTop3(self.blockTop2(self.blockTop1(top))))

        features_front = torch.flatten(features_front, start_dim=1)
        features_left = torch.flatten(features_left, start_dim=1)
        features_back = torch.flatten(features_back, start_dim=1)
        features_right = torch.flatten(features_right, start_dim=1)
        features_top = torch.flatten(features_top, start_dim=1)

        features = torch.cat([features_front, features_left, features_back, features_right, features_top], dim=1)

        latent_code = self.ln3(self.ln2(self.ln1(features)))

        return latent_code    

code = torch.empty(10,16)
xyz_coord = torch.empty(10,3)

decoder = DecoderSDF(16)

decoder(code,xyz_coord)
pytorch_total_params = sum(p.numel() for p in decoder.parameters())
print("Decoder parameters: {}".format(pytorch_total_params))


# image = torch.empty(10,3,300,450)
# loc = torch.empty(10,20)

# encoder = EncoderSDF(16)

# encoder(image,loc)
# pytorch_total_params = sum(p.numel() for p in encoder.parameters())
# print("Encoder parameters: {}".format(pytorch_total_params))



grid = torch.empty(10,3,50,25,25)

encoder = EncoderGrid(16)

encoder(grid)
pytorch_total_params = sum(p.numel() for p in encoder.parameters())
print("Encoder grid parameter: {}".format(pytorch_total_params))


encoder = EncoderGrid2(16)

encoder(grid)
pytorch_total_params = sum(p.numel() for p in encoder.parameters())
print("Encoder grid 2 parameter: {}".format(pytorch_total_params))


width, height, depth = 64, 64, 128

front = torch.empty(10,3,width,height)
left = torch.empty(10,3,depth,height)
back = torch.empty(10,3,width,height)
right = torch.empty(10,3,depth,height)
top = torch.empty(10,3,width,depth)

encoder = EncoderFace(16)

encoder(front, left, back, right, top)
pytorch_total_params = sum(p.numel() for p in encoder.parameters())
print("Encoder face parameters: {}".format(pytorch_total_params))
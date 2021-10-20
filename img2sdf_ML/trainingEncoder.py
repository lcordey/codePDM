import h5py
import math
import numpy as np
from numpy.core.fromnumeric import transpose
import torch
import torch.nn as nn
import pickle
import argparse
import os
import IPython
import glob
import imageio
import random
import time

from networks import DecoderSDF, EncoderSDF, EncoderGrid
from dataLoader import DatasetGrid
from marching_cubes_rgb import *

###### parameter #####

DECODER_PATH = "models_pth/decoderSDF.pth"
ENCODER_PATH = "models_pth/encoderSDF.pth"
LATENT_VECS_TARGET_PATH = "models_pth/latent_vecs_target.pth"
LATENT_VECS_PRED_PATH = "models_pth/latent_vecs_pred.pth"

ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"


num_epoch = 100
batch_size = 10

eta_encoder = 1e-4
gammaLR = 0.99

# ratio_image_used = 0.5

height_input_image = 300
width_input_image = 450

num_slices = 50
width_input_network = 25
height_input_network = 25


# width_input_network = 68
# height_input_network = 68
# depth_input_network = 120

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def initialize_dataset():
    list_scene = list(annotations.keys())
    dict_scene_2_code = dict()

    for scene_hash, scene_id in zip(list_scene, range(len(list_scene))):
        dict_scene_2_code[scene_hash] = scene_id

    return list_scene, dict_scene_2_code


decoder = torch.load(DECODER_PATH).cuda()
target_vecs = torch.load(LATENT_VECS_TARGET_PATH).cuda()

annotations_file = open(ANNOTATIONS_PATH, "rb")
annotations = pickle.load(annotations_file)

num_image_per_scene = len(annotations[next(iter(annotations.keys()))])
num_scene, latent_size = target_vecs.shape
assert(num_scene == len(annotations.keys()))

params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': False
          }

list_scene, dict_scene_2_code = initialize_dataset()

list_scene = np.repeat(list_scene, 100)

training_set_grid = DatasetGrid(list_scene,
                       dict_scene_2_code,
                       target_vecs.cpu(),
                       annotations,
                       0,
                       num_image_per_scene,
                       width_input_image,
                       height_input_image,
                       num_slices,
                       width_input_network,
                       height_input_network)
training_generator_grid = torch.utils.data.DataLoader(training_set_grid, **params)


# encoder
# encoder = EncoderSDF(latent_size).cuda()
encoder = EncoderGrid(latent_size).cuda()
# encoder = EncoderFace(latent_size).cuda()

encoder.apply(init_weights)

# encoder = torch.load(ENCODER_PATH).cuda()

loss = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
        [
            {
                "params": encoder.parameters(),
                "lr": eta_encoder,
                "eps": 1e-8,
            },
        ]
    )

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gammaLR)

####################### Training loop ##########################
log_loss = []


start_time = time.time()


encoder.train()
for epoch in range(num_epoch):


    for batch_input_im, batch_target_code in training_generator_grid:
    # for batch_front, batch_left, batch_back, batch_right, batch_top, batch_target_code in training_generator_grid:


        # print(f"total time: {time.time() - start_time}")
        # start_time = time.time()

        optimizer.zero_grad()

        input_im, target_code = batch_input_im.cuda(), batch_target_code.cuda()
        # front, left, back, right, top, target_code = batch_front.cuda(), batch_left.cuda(), batch_back.cuda(), batch_right.cuda(), batch_top.cuda(), batch_target_code.cuda()

        pred_vecs = encoder(input_im)
        # pred_vecs = encoder(front, left, back, right, top)

        loss_pred = loss(pred_vecs, target_code)
        log_loss.append(loss_pred.detach().cpu())

        #update weights
        loss_pred.backward()
        optimizer.step()

        print("epoch: {}, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}".format(epoch, torch.Tensor(log_loss[-10:]).mean(), \
        abs(pred_vecs - target_code).mean(), abs(pred_vecs).mean(), abs(target_code).mean(), optimizer.param_groups[0]['lr']  ))

        # print(f"network time: {time.time() - start_time}")

    
    scheduler.step()


####################### Evaluation ##########################

# encoder.eval()

# print("******************** VALIDATION ********************")
# print("L2 loss: {:.5f}, L1 loss: {:.5f} norm_pred: {:.5f}, norm target: {:.5f}".format(loss_pred.mean().item(), \
#     abs(pred_vecs - target_vecs.unsqueeze(1).repeat(1, num_validation_image_per_scene, 1)).mean(), abs(pred_vecs).mean(), abs(target_vecs[batch_scene_idx]).mean()))


print(f"time for training: {time.time() - start_time}")


#save model
# torch.save(pred_vecs.detach().cpu(), LATENT_VECS_PRED_PATH)
torch.save(encoder, ENCODER_PATH)

#save logs plot
avrg_loss = []
for i in range(0,len(log_loss)):
    avrg_loss.append(torch.Tensor(log_loss[i-20:i]).mean())
    

from matplotlib import pyplot as plt
plt.figure()
plt.title("Total loss")
plt.semilogy(avrg_loss[:])
plt.savefig("../../image2sdf/logs/log_total")

with open("../../image2sdf/logs/log.txt", "wb") as fp:
    pickle.dump(avrg_loss, fp)


# IPython.embed()
from typing import NewType
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

from networks import DecoderSDF, EncoderSDF, EncoderGrid, EncoderGrid2, EncoderFace
from dataLoader import DatasetGrid, DatasetFace
from marching_cubes_rgb import *

###### parameter #####

DECODER_PATH = "models_pth/decoderSDF.pth"
ENCODER_GRID_PATH = "models_pth/encoderGrid.pth"
ENCODER_FACE_PATH = "models_pth/encoderFace.pth"
LATENT_VECS_TARGET_PATH = "models_pth/latent_vecs_target.pth"
LATENT_VECS_PRED_PATH = "models_pth/latent_vecs_pred.pth"

ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"

NEWTORK = 'grid'
# NEWTORK = 'face'

num_epoch = 1
batch_size = 10

eta_encoder = 1e-4
gammaLR = 0.9

num_scene_validation = 15

height_input_image = 300
width_input_image = 450

num_slices = 48

width_input_network_grid = 24
height_input_network_grid = 24

width_input_network_face = 64
height_input_network_face = 64

depth_input_network = 128

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def initialize_dataset(annotations):
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

num_scene_training = num_scene -num_scene_validation
total_model_to_show = num_scene_training * num_image_per_scene * num_epoch

params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': False
          }

list_scene, dict_scene_2_code = initialize_dataset(annotations)

list_scene_training = list_scene[:-num_scene_validation]
list_scene_validation = list_scene[-num_scene_validation:]

list_scene_training = np.repeat(list_scene_training, num_image_per_scene)

training_set_grid = DatasetGrid(list_scene_training,
                       dict_scene_2_code,
                       target_vecs.cpu(),
                       annotations,
                       0,
                       num_image_per_scene,
                       width_input_image,
                       height_input_image,
                       num_slices,
                       width_input_network_grid,
                       height_input_network_grid)

training_generator_grid = torch.utils.data.DataLoader(training_set_grid, **params)


training_set_face = DatasetFace(list_scene_training,
                       dict_scene_2_code,
                       target_vecs.cpu(),
                       annotations,
                       0,
                       num_image_per_scene,
                       width_input_image,
                       height_input_image,
                       width_input_network_face,
                       height_input_network_face,
                       depth_input_network)

training_generator_face = torch.utils.data.DataLoader(training_set_face, **params)


validation_set_grid = DatasetGrid(list_scene_validation,
                       dict_scene_2_code,
                       target_vecs.cpu(),
                       annotations,
                       0,
                       num_image_per_scene,
                       width_input_image,
                       height_input_image,
                       num_slices,
                       width_input_network_grid,
                       height_input_network_grid)

validation_generator_grid = torch.utils.data.DataLoader(validation_set_grid, **params)


validation_set_face = DatasetFace(list_scene_validation,
                       dict_scene_2_code,
                       target_vecs.cpu(),
                       annotations,
                       0,
                       num_image_per_scene,
                       width_input_image,
                       height_input_image,
                       width_input_network_face,
                       height_input_network_face,
                       depth_input_network)

validation_generator_face = torch.utils.data.DataLoader(validation_set_face, **params)

# encoder
# encoder = EncoderSDF(latent_size).cuda()
# encoder = EncoderGrid(latent_size).cuda()

if NEWTORK == 'grid':
    encoder = EncoderGrid2(latent_size).cuda()
elif NEWTORK == 'face':
    encoder = EncoderFace(latent_size).cuda()

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
log_loss_validation = []
log_loss_sdf_validation = []
log_loss_rgb_validation = []

resolution = 64

xyz = torch.empty(resolution * resolution * resolution, 3).cuda()
for x in range(resolution):
    for y in range(resolution):
        for z in range(resolution):
            xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])


time_start = time.time()


encoder.train()
print("Start trainging...")

if NEWTORK == 'grid':
    for epoch in range(num_epoch):
        count_model = 0
        for batch_input_im, batch_target_code in training_generator_grid:
            optimizer.zero_grad()

            input_im, target_code = batch_input_im.cuda(), batch_target_code.cuda()
            pred_vecs = encoder(input_im)

            loss_pred = loss(pred_vecs, target_code)
            log_loss.append(loss_pred.detach().cpu())

            #update weights
            loss_pred.backward()
            optimizer.step()

            # print(f"network time: {time.time() - time_start}")

            time_passed = time.time() - time_start
            model_seen = len(log_loss) * batch_size
            time_per_model = time_passed/(model_seen)
            time_left = time_per_model * (total_model_to_show - model_seen)
            # print("epoch: {}/{}, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}, time left: {} min".format(epoch, count_model, torch.Tensor(log_loss[-10:]).mean(), \
            # abs(pred_vecs - target_code).mean(), abs(pred_vecs).mean(), abs(target_code).mean(), optimizer.param_groups[0]['lr'],  (int)(time_left/60) ))

            if count_model%(total_model_to_show/num_epoch/100) == 0:
                print("epoch: {}/{}, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}, time left: {} min".format(epoch, count_model, torch.Tensor(log_loss[-10:]).mean(), \
                abs(pred_vecs - target_code).mean(), abs(pred_vecs).mean(), abs(target_code).mean(), optimizer.param_groups[0]['lr'],  (int)(time_left/60) ))

            if count_model%(total_model_to_show/num_epoch/10) == 0:
                encoder.eval()
                loss_pred_validation = []
                loss_sdf_validation = []
                loss_rgb_validation = []
                for batch_input_im_validation, batch_target_code_validation in validation_generator_grid:
                    input_im_validation, target_code_validation = batch_input_im_validation.cuda(), batch_target_code_validation.cuda()
                    pred_vecs_validation = encoder(input_im_validation)
                    loss_pred_validation.append(loss(pred_vecs_validation, target_code_validation).detach().cpu())

                    sdf_validation = decoder(pred_vecs_validation)
                    sdf_target= decoder(target_code_validation)

                    # assign weight of 0 for easy samples that are well trained
                    threshold_precision = 1/resolution
                    weight_sdf = ~((sdf_validation[:,0] > threshold_precision).squeeze() * (sdf_target[:,0] > threshold_precision).squeeze()) \
                        * ~((sdf_validation[:,0] < -threshold_precision).squeeze() * (sdf_target[:,0] < -threshold_precision).squeeze())


                    #L1 loss, only for hard samples
                    loss_sdf = loss(sdf_validation[:,0].squeeze(), sdf_target[:,0])
                    loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()

                    # loss rgb
                    lambda_rgb = 1/100
                    
                    rgb_gt_normalized = sdf_target[:,1:]/255
                    loss_rgb = loss(sdf_validation[:,1:], rgb_gt_normalized)
                    loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean() * weight_sdf.numel()/weight_sdf.count_nonzero() * lambda_rgb
        
                    loss_sdf_validation.append(loss_sdf)
                    loss_rgb_validation.append(loss_rgb)
                
                loss_pred_validation = torch.tensor(loss_pred_validation).mean()
                loss_sdf_validation = torch.tensor(loss_sdf_validation).mean()
                loss_rgb_validation = torch.tensor(loss_rgb_validation).mean()
                print("\n********** VALIDATION **********")
                print(f"validation L2 loss: {loss_pred_validation}\n")
                print(f"validation sdf loss: {loss_sdf_validation}\n")
                print(f"validation rgb loss: {loss_rgb_validation}\n")
                log_loss_validation.append(loss_pred_validation)
                log_loss_sdf_validation.append(loss_sdf_validation)
                log_loss_rgb_validation.append(loss_rgb_validation)

                encoder.train()


                count_model += batch_size
                
        scheduler.step()


elif NEWTORK == 'face':
    for epoch in range(num_epoch):
        count_model = 0
        for batch_front, batch_left, batch_back, batch_right, batch_top, batch_target_code in training_generator_face:
            optimizer.zero_grad()

            front, left, back, right, top, target_code = batch_front.cuda(), batch_left.cuda(), batch_back.cuda(), batch_right.cuda(), batch_top.cuda(), batch_target_code.cuda()
            pred_vecs = encoder(front, left, back, right, top)

            loss_pred = loss(pred_vecs, target_code)
            log_loss.append(loss_pred.detach().cpu())

            #update weights
            loss_pred.backward()
            optimizer.step()

            time_passed = time.time() - time_start
            model_seen = len(log_loss) * batch_size
            time_per_model = time_passed/(model_seen)
            time_left = time_per_model * (total_model_to_show - model_seen)
            count_model += batch_size
            # print("epoch: {}/{}, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}, time left: {} min".format(epoch, count_model, torch.Tensor(log_loss[-10:]).mean(), \
            # abs(pred_vecs - target_code).mean(), abs(pred_vecs).mean(), abs(target_code).mean(), optimizer.param_groups[0]['lr'],  (int)(time_left/60) ))


            if count_model%(total_model_to_show/num_epoch/100) == 0:
                print("epoch: {}/{}, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}, time left: {} min".format(epoch, count_model, torch.Tensor(log_loss[-10:]).mean(), \
                abs(pred_vecs - target_code).mean(), abs(pred_vecs).mean(), abs(target_code).mean(), optimizer.param_groups[0]['lr'],  (int)(time_left/60) ))

            if count_model%(total_model_to_show/num_epoch/10) == 0:
            
                encoder.eval()
                loss_pred_validation = []
                for batch_front, batch_left, batch_back, batch_right, batch_top, batch_target_code in validation_generator_face:
                    front, left, back, right, top, target_code = batch_front.cuda(), batch_left.cuda(), batch_back.cuda(), batch_right.cuda(), batch_top.cuda(), batch_target_code.cuda()
                    pred_vecs = encoder(front, left, back, right, top)

                    loss_pred_validation.append(loss(pred_vecs, target_code).detach().cpu())
                
                loss_validation = torch.tensor(loss_pred_validation).mean()
                print("\n********** VALIDATION **********")
                print(f"validation L2 loss: {loss_validation}\n")
                log_loss_validation.append(loss_validation)

                encoder.train()


        scheduler.step()



####################### Evaluation ##########################

# encoder.eval()

# print("******************** VALIDATION ********************")
# print("L2 loss: {:.5f}, L1 loss: {:.5f} norm_pred: {:.5f}, norm target: {:.5f}".format(loss_pred.mean().item(), \
#     abs(pred_vecs - target_vecs.unsqueeze(1).repeat(1, num_validation_image_per_scene, 1)).mean(), abs(pred_vecs).mean(), abs(target_vecs[batch_scene_idx]).mean()))


print(f"time for training: {(int)((time.time() - time_start)/60)}")


#save model
# torch.save(pred_vecs.detach().cpu(), LATENT_VECS_PRED_PATH)
if NEWTORK == 'grid':
    torch.save(encoder, ENCODER_GRID_PATH)
else:
    torch.save(encoder, ENCODER_FACE_PATH)

#save logs plot
avrg_loss = []
for i in range(0,len(log_loss)):
    avrg_loss.append(torch.Tensor(log_loss[i-20:i]).mean())

    

from matplotlib import pyplot as plt
plt.figure()
plt.title("Training loss")
plt.xlabel("Number of images shown")
plt.ylabel("L2 loss")
plt.semilogy(np.arange(len(avrg_loss)) * batch_size, avrg_loss[:], label = "training loss")
plt.savefig("../../image2sdf/logs/log_total")

plt.figure()
plt.title("Latent code loss Validation")
plt.xlabel("Number of images shown")
plt.ylabel("L2 loss")
plt.semilogy(np.arange(len(log_loss_validation)) * (total_model_to_show/num_epoch/10), log_loss_validation[:], label = "validation loss")
plt.savefig("../../image2sdf/logs/log_total_validation")

plt.figure()
plt.title("Loss sdf")
plt.xlabel("Number of images shown")
plt.ylabel("L2 loss")
plt.semilogy(np.arange(len(log_loss_sdf_validation)) * (total_model_to_show/num_epoch/10), log_loss_sdf_validation[:], label = "validation loss sdf")
plt.savefig("../../image2sdf/logs/log_total_validation")

plt.figure()
plt.title("Loss rgb")
plt.xlabel("Number of images shown")
plt.ylabel("L2 loss")
plt.semilogy(np.arange(len(log_loss_rgb_validation)) * (total_model_to_show/num_epoch/10), log_loss_rgb_validation[:], label = "validation loss rgb")
plt.savefig("../../image2sdf/logs/log_total_validation")

with open("../../image2sdf/logs/log.txt", "wb") as fp:
    pickle.dump(avrg_loss, fp)


# IPython.embed()
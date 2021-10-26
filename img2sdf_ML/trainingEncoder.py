from typing import NewType
import h5py
import math
import numpy as np
from numpy.core.defchararray import count
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

def cosine_distance(a,b):
    return a.dot(b)/(a.norm() * b.norm())


decoder = torch.load(DECODER_PATH).cuda()
decoder.eval()
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


params_validation = {'batch_size': 1,
          'shuffle': False,
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

validation_generator_grid = torch.utils.data.DataLoader(validation_set_grid, **params_validation)


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

validation_generator_face = torch.utils.data.DataLoader(validation_set_face, **params_validation)

# encoder
if NEWTORK == 'grid':
    encoder = EncoderGrid2(latent_size).cuda()
elif NEWTORK == 'face':
    encoder = EncoderFace(latent_size).cuda()

encoder.apply(init_weights)

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

resolution = 64

xyz = torch.empty(resolution * resolution * resolution, 3).cuda()
for x in range(resolution):
    for y in range(resolution):
        for z in range(resolution):
            xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])


time_start = time.time()

log_loss = []

log_same_model_cos = []
log_diff_model_cos = []
log_same_model_l2 = []
log_diff_model_l2 = []

log_loss_pred_validation = []
log_cosine_distance_validation = []

log_loss_sdf_validation = []
log_loss_rgb_validation = []

encoder.train()
print("Start trainging...")

if NEWTORK == 'grid':
    for epoch in range(num_epoch):
        count_model = 0
        for batch_input_im, batch_target_code in training_generator_grid:
            if count_model > total_model_to_show/num_epoch/20:
                break

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
          
            count_model += batch_size

            if count_model%(total_model_to_show/num_epoch/1000) == 0:
                print("epoch: {}/{}, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}, time left: {} min".format(epoch, count_model, torch.Tensor(log_loss[-10:]).mean(), \
                abs(pred_vecs - target_code).mean(), abs(pred_vecs).mean(), abs(target_code).mean(), optimizer.param_groups[0]['lr'],  (int)(time_left/60) ))


            # if count_model%(total_model_to_show/num_epoch/1000) == 0:
            #     print("epoch: {}/{}, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}, time left: {} min".format(epoch, count_model, torch.Tensor(log_loss[-10:]).mean(), \
            #     abs(pred_vecs - target_code).mean(), abs(pred_vecs).mean(), abs(target_code).mean(), optimizer.param_groups[0]['lr'],  (int)(time_left/60) ))

            #     sdf_validation = decoder(pred_vecs.repeat_interleave(resolution * resolution * resolution, dim=0),xyz).detach()
            #     sdf_target= decoder(target_code.repeat_interleave(resolution * resolution * resolution, dim=0),xyz).detach()

            #     # assign weight of 0 for easy samples that are well trained
            #     threshold_precision = 1/resolution
            #     weight_sdf = ~((sdf_validation[:,0] > threshold_precision).squeeze() * (sdf_target[:,0] > threshold_precision).squeeze()) \
            #         * ~((sdf_validation[:,0] < -threshold_precision).squeeze() * (sdf_target[:,0] < -threshold_precision).squeeze())


            #     #L1 loss, only for hard samples
            #     loss_sdf = torch.nn.MSELoss(reduction='none')(sdf_validation[:,0].squeeze(), sdf_target[:,0])
            #     loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()


            #     # IPython.embed()
            
            #     # loss rgb
            #     lambda_rgb = 1/100
                
            #     rgb_gt_normalized = sdf_target[:,1:]
            #     loss_rgb = torch.nn.MSELoss(reduction='none')(sdf_validation[:,1:], rgb_gt_normalized)
            #     loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean() * weight_sdf.numel()/weight_sdf.count_nonzero() * lambda_rgb
    
            #     print("\n********** VALIDATION **********")
            #     print(f"validation latent code l2 loss: {loss(pred_vecs, target_code)}")
            #     print(f"validation sdf loss: {loss_sdf}")
            #     print(f"validation rgb loss: {loss_rgb}")
            #     print("\n")
            #     log_loss_validation.append(loss_pred)
            #     log_loss_sdf_validation.append(loss_sdf)
            #     log_loss_rgb_validation.append(loss_rgb)


            if count_model%(total_model_to_show/num_epoch/100) == 0 or count_model == batch_size:
                encoder.eval()
                num_epoch_validation = 5
                pred_vecs_matrix = torch.empty([num_scene_validation,num_epoch_validation, latent_size])
                loss_pred_validation = []
                cosine_distance_validation = []
                loss_sdf_validation = []
                loss_rgb_validation = []

                
                for epoch_validation in range(num_epoch_validation):
                    scene_id = 0
                    for batch_input_im_validation, batch_target_code_validation in validation_generator_grid:
                        input_im_validation, target_code_validation = batch_input_im_validation.cuda(), batch_target_code_validation.cuda()
                        pred_vecs_validation = encoder(input_im_validation).detach()

                        pred_vecs_matrix[scene_id, epoch_validation, :] = pred_vecs_validation
                        # loss_pred_validation.append(loss(pred_vecs_validation, target_code_validation))
                        loss_pred_validation.append(torch.norm(pred_vecs_validation - target_code_validation))
                        cosine_distance_validation.append(cosine_distance(pred_vecs_validation.squeeze(), target_code_validation.squeeze()))

                        if scene_id == 0:
                            sdf_validation = decoder(pred_vecs_validation.repeat_interleave(resolution * resolution * resolution, dim=0),xyz).detach()
                            sdf_target= decoder(target_code_validation.repeat_interleave(resolution * resolution * resolution, dim=0),xyz).detach()

                            # assign weight of 0 for easy samples that are well trained
                            threshold_precision = 1/resolution
                            weight_sdf = ~((sdf_validation[:,0] > threshold_precision).squeeze() * (sdf_target[:,0] > threshold_precision).squeeze()) \
                                * ~((sdf_validation[:,0] < -threshold_precision).squeeze() * (sdf_target[:,0] < -threshold_precision).squeeze())

                            #L2 loss, only for hard samples
                            loss_sdf = torch.nn.MSELoss(reduction='none')(sdf_validation[:,0].squeeze(), sdf_target[:,0])
                            loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()
                        
                            # loss rgb
                            lambda_rgb = 1/100
                            
                            rgb_gt_normalized = sdf_target[:,1:]
                            loss_rgb = torch.nn.MSELoss(reduction='none')(sdf_validation[:,1:], rgb_gt_normalized)
                            loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean() * weight_sdf.numel()/weight_sdf.count_nonzero() * lambda_rgb
                
                            loss_sdf_validation.append(loss_sdf)
                            loss_rgb_validation.append(loss_rgb)


                        scene_id += 1
                
                cosine_distance_validation = torch.tensor(cosine_distance_validation).mean()
                loss_pred_validation = torch.tensor(loss_pred_validation).mean()
                loss_sdf_validation = torch.tensor(loss_sdf_validation).mean()
                loss_rgb_validation = torch.tensor(loss_rgb_validation).mean()


                similarity_same_model_cos = []
                similarity_different_model_cos = []

                similarity_same_model_l2 = []
                similarity_different_model_l2 = []

                for scene_id_1 in range(num_scene_validation):
                    for scene_id_2 in range(scene_id_1, num_scene_validation):
                        for vec1 in range(num_epoch_validation):
                            for vec2 in range(num_epoch_validation):
                                dist = cosine_distance(pred_vecs_matrix[scene_id_1,vec1,:], pred_vecs_matrix[scene_id_2,vec2,:])
                                # l2 = loss(pred_vecs_matrix[scene_id_1,vec1,:], pred_vecs_matrix[scene_id_2,vec2,:])
                                l2 = torch.norm(pred_vecs_matrix[scene_id_1,vec1,:]- pred_vecs_matrix[scene_id_2,vec2,:])
                                if scene_id_1 == scene_id_2 and vec2 != vec1:
                                    similarity_same_model_cos.append(dist)
                                    similarity_same_model_l2.append(l2)
                                elif scene_id_1 != scene_id_2:
                                    similarity_different_model_cos.append(dist)
                                    similarity_different_model_l2.append(l2)


                same_model_cos = torch.tensor(similarity_same_model_cos).mean()
                diff_model_cos = torch.tensor(similarity_different_model_cos).mean()
                same_model_l2 = torch.tensor(similarity_same_model_l2).mean()
                diff_model_l2 = torch.tensor(similarity_different_model_l2).mean()

                print("\n********** VALIDATION **********")

                print(f"average cosinus distance between same models : {same_model_cos}")
                print(f"average cosinus distance between differents models: {diff_model_cos}")

                print(f"average l2 distance between same models: {same_model_l2}")
                print(f"average l2 distance between differents models: {diff_model_l2}")

                print(f"avarage cosinus distance with target: {cosine_distance_validation}")
                print(f"average L2 distance with target: {loss_pred_validation}")

                print(f"average reconstruction sdf loss: {loss_sdf_validation}")
                print(f"average reconstruction rgb loss: {loss_rgb_validation}")
                print("\n")

                log_same_model_cos.append(same_model_cos)
                log_diff_model_cos.append(diff_model_cos)
                log_same_model_l2.append(same_model_l2)
                log_diff_model_l2.append(diff_model_l2)
                log_cosine_distance_validation.append(cosine_distance_validation)
                log_loss_pred_validation.append(loss_pred_validation)
                log_loss_sdf_validation.append(loss_sdf_validation)
                log_loss_rgb_validation.append(loss_rgb_validation)

                encoder.train()


                
        scheduler.step()




# elif NEWTORK == 'face':
#     for epoch in range(num_epoch):
#         count_model = 0
#         for batch_front, batch_left, batch_back, batch_right, batch_top, batch_target_code in training_generator_face:
#             optimizer.zero_grad()

#             front, left, back, right, top, target_code = batch_front.cuda(), batch_left.cuda(), batch_back.cuda(), batch_right.cuda(), batch_top.cuda(), batch_target_code.cuda()
#             pred_vecs = encoder(front, left, back, right, top)

#             loss_pred = loss(pred_vecs, target_code)
#             log_loss.append(loss_pred.detach().cpu())

#             #update weights
#             loss_pred.backward()
#             optimizer.step()

#             time_passed = time.time() - time_start
#             model_seen = len(log_loss) * batch_size
#             time_per_model = time_passed/(model_seen)
#             time_left = time_per_model * (total_model_to_show - model_seen)
#             count_model += batch_size

#             if count_model%(total_model_to_show/num_epoch/100) == 0:
#                 print("epoch: {}/{}, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}, time left: {} min".format(epoch, count_model, torch.Tensor(log_loss[-10:]).mean(), \
#                 abs(pred_vecs - target_code).mean(), abs(pred_vecs).mean(), abs(target_code).mean(), optimizer.param_groups[0]['lr'],  (int)(time_left/60) ))

#             if count_model%(total_model_to_show/num_epoch/10) == 0:
            
#                 encoder.eval()
#                 loss_pred_validation = []
#                 for batch_front, batch_left, batch_back, batch_right, batch_top, batch_target_code in validation_generator_face:
#                     front, left, back, right, top, target_code = batch_front.cuda(), batch_left.cuda(), batch_back.cuda(), batch_right.cuda(), batch_top.cuda(), batch_target_code.cuda()
#                     pred_vecs = encoder(front, left, back, right, top)

#                     loss_pred_validation.append(loss(pred_vecs, target_code).detach().cpu())
                
#                 loss_validation = torch.tensor(loss_pred_validation).mean()
#                 print("\n********** VALIDATION **********")
#                 print(f"validation L2 loss: {loss_validation}\n")
#                 log_loss_pred_validation.append(loss_validation)

#                 encoder.train()
#         scheduler.step()


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
plt.title("Latent code cosine distance Validation")
plt.xlabel("Number of images shown")
plt.ylabel("cosine distance")
plt.plot(np.arange(len(log_same_model_cos)) * (total_model_to_show/num_epoch/100), log_same_model_cos[:], label = "same models")
plt.plot(np.arange(len(log_diff_model_cos)) * (total_model_to_show/num_epoch/100), log_diff_model_cos[:], label = "differents models")
plt.plot(np.arange(len(log_cosine_distance_validation)) * (total_model_to_show/num_epoch/100), log_cosine_distance_validation[:], label = "target and prediction")
plt.legend()
plt.savefig("../../image2sdf/logs/log_cosine_distance_validation")


plt.figure()
plt.title("Latent code l2 distance Validation")
plt.xlabel("Number of images shown")
plt.ylabel("l2 distance")
plt.plot(np.arange(len(log_same_model_l2)) * (total_model_to_show/num_epoch/100), log_same_model_l2[:], label = "same models")
plt.plot(np.arange(len(log_diff_model_l2)) * (total_model_to_show/num_epoch/100), log_diff_model_l2[:], label = "differents models")
plt.semilogy(np.arange(len(log_loss_pred_validation)) * (total_model_to_show/num_epoch/100), log_loss_pred_validation[:], label = "target and prediction")
plt.legend()
plt.savefig("../../image2sdf/logs/log_l2_distance_validation")


plt.figure()
plt.title("Loss sdf")
plt.xlabel("Number of images shown")
plt.ylabel("L2 loss")
plt.semilogy(np.arange(len(log_loss_sdf_validation)) * (total_model_to_show/num_epoch/1000), log_loss_sdf_validation[:], label = "validation loss sdf")
plt.savefig("../../image2sdf/logs/log_sdf_validation")

plt.figure()
plt.title("Loss rgb")
plt.xlabel("Number of images shown")
plt.ylabel("L2 loss")
plt.semilogy(np.arange(len(log_loss_rgb_validation)) * (total_model_to_show/num_epoch/1000), log_loss_rgb_validation[:], label = "validation loss rgb")
plt.savefig("../../image2sdf/logs/log_rgb_validation")

with open("../../image2sdf/logs/log.txt", "wb") as fp:
    pickle.dump(avrg_loss, fp)

# with open("../../image2sdf/logs/log_cos.txt", "wb") as fp:
#     pickle.dump(log_same_model_cos, fp)

# with open("../../image2sdf/logs/log_l2.txt", "wb") as fp:
#     pickle.dump(log_same_model_l2, fp)


# IPython.embed()
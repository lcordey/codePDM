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

from networks import DecoderSDF, EncoderSDF
from marching_cubes_rgb import *

###### parameter #####

TESTING = False

DECODER_PATH = "models_pth/decoderSDF.pth"
ENCODER_PATH = "models_pth/encoderSDF.pth"
LATENT_VECS_TARGET_PATH = "models_pth/latent_vecs_target.pth"
LATENT_VECS_PRED_PATH = "models_pth/latent_vecs_pred.pth"

ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"

num_epoch = 100000
batch_size = 25

eta_encoder = 5e-4
gammaLR = 0.9999


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


decoder = torch.load(DECODER_PATH).cuda()
target_vecs = torch.load(LATENT_VECS_TARGET_PATH).cuda()

annotations_file = open(ANNOTATIONS_PATH, "rb")
annotations = pickle.load(annotations_file)

num_image_per_scene = len(annotations[next(iter(annotations.keys()))])
num_scene, latent_size = target_vecs.shape
assert(num_scene == len(annotations.keys()))

input_images = None
input_locations = np.empty([num_scene, num_image_per_scene, 20])

for scene, scene_id in zip(annotations.keys(), range(num_scene)):
    for image, image_id in zip(glob.glob(IMAGES_PATH + scene + '/*'), range(num_image_per_scene)):

        # save image
        im = imageio.imread(image)

        if input_images is None:
            height = im.shape[0]
            width = im.shape[1]

            input_images = np.empty([num_scene, num_image_per_scene, im.shape[2], im.shape[0], im.shape[1]])

        input_images[scene_id, image_id, :,:,:] = np.transpose(im,(2,0,1))

        # save locations
        for loc, loc_id in zip(annotations[scene][image_id].keys(), range(len(annotations[scene][image_id].keys()))):
            if loc[-1] == 'x' or loc[-5:] == 'width':
                input_locations[scene_id, image_id, loc_id] = annotations[scene][image_id][loc]/width
            else:
                input_locations[scene_id, image_id, loc_id] = annotations[scene][image_id][loc]/height

input_locations = input_locations - 0.5
input_images = input_images/255 - 0.5

ratio_training_validation = 0.8
num_training_image_per_scene = (np.int)(np.round(num_image_per_scene * ratio_training_validation))
num_validation_image_per_scene = num_image_per_scene - num_training_image_per_scene

train_images_idx = np.arange(num_training_image_per_scene)
validation_images_idx = np.arange(num_training_image_per_scene, num_image_per_scene)

train_input_im = torch.tensor(input_images[:,train_images_idx,:,:,:], dtype = torch.float).cuda()
validation_input_im = torch.tensor(input_images[:,validation_images_idx,:,:,:], dtype = torch.float).cuda()
train_input_loc = torch.tensor(input_locations[:,train_images_idx,:], dtype = torch.float).cuda()
validation_input_loc = torch.tensor(input_locations[:,validation_images_idx,:], dtype = torch.float).cuda()


# encoder
encoder = EncoderSDF(latent_size).cuda()
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
log_loss = []

encoder.train()
for epoch in range(num_epoch):

    optimizer.zero_grad()

    # random batch
    batch_scene_idx = np.random.randint(num_scene, size = batch_size)
    batch_image_idx = np.random.randint(num_training_image_per_scene, size = batch_size)

    ### all samples together
    # batch_scene_idx = np.repeat(np.arange(num_scene), num_training_image_per_scene)
    # batch_image_idx = np.tile(np.arange(num_training_image_per_scene), num_scene)

    pred_vecs = encoder(train_input_im[batch_scene_idx, batch_image_idx, :, :, :], train_input_loc[batch_scene_idx,batch_image_idx, :])

    loss_pred = loss(pred_vecs, target_vecs[batch_scene_idx])
    log_loss.append(loss_pred.detach().cpu())

    #update weights
    loss_pred.backward()
    optimizer.step()
    scheduler.step()

    print("epoch: {}, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}".format(epoch, torch.Tensor(log_loss[-10:]).mean(), \
        abs(pred_vecs - target_vecs[batch_scene_idx]).mean(), abs(pred_vecs).mean(), abs(target_vecs[batch_scene_idx]).mean(), optimizer.param_groups[0]['lr']  ))



####################### Evaluation ##########################

encoder.eval()

### evaluation on training data
# batch_image_idx = np.arange(num_training_image_per_scene)
# pred_vecs = torch.empty([num_scene, num_training_image_per_scene, latent_size]).cuda()

# for scene_id in range(num_scene):
#     pred_vecs[scene_id,:,:] = encoder(train_input_im[scene_id, batch_image_idx, :, :, :], train_input_loc[scene_id,batch_image_idx, :]).detach()

# batch_scene_idx = np.repeat(np.arange(num_scene),num_training_image_per_scene)
# loss_pred = loss(pred_vecs, target_vecs.unsqueeze(1).repeat(1, num_training_image_per_scene, 1))


# print("******************** VALIDATION ********************")
# print("L2 loss: {:.5f}, L1 loss: {:.5f} norm_pred: {:.5f}, norm target: {:.5f}".format(loss_pred.mean().item(), \
#     abs(pred_vecs - target_vecs.unsqueeze(1).repeat(1, num_training_image_per_scene, 1)).mean(), abs(pred_vecs).mean(), abs(target_vecs[batch_scene_idx]).mean()))


### evaluation on validation data
batch_image_idx = np.arange(num_validation_image_per_scene)
pred_vecs = torch.empty([num_scene, num_validation_image_per_scene, latent_size]).cuda()

for scene_id in range(num_scene):
    pred_vecs[scene_id,:,:] = encoder(validation_input_im[scene_id, batch_image_idx, :, :, :], validation_input_loc[scene_id,batch_image_idx, :]).detach()

batch_scene_idx = np.repeat(np.arange(num_scene),num_validation_image_per_scene)
loss_pred = loss(pred_vecs, target_vecs.unsqueeze(1).repeat(1, num_validation_image_per_scene, 1))


print("******************** VALIDATION ********************")
print("L2 loss: {:.5f}, L1 loss: {:.5f} norm_pred: {:.5f}, norm target: {:.5f}".format(loss_pred.mean().item(), \
    abs(pred_vecs - target_vecs.unsqueeze(1).repeat(1, num_validation_image_per_scene, 1)).mean(), abs(pred_vecs).mean(), abs(target_vecs[batch_scene_idx]).mean()))



#save model
torch.save(pred_vecs.detach().cpu(), LATENT_VECS_PRED_PATH)
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
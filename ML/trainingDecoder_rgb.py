import h5py
import math
import numpy as np
import torch

from decoderSDF_rgb import DecoderSDF
from marching_cubes_rgb import *

###### parameter #####

TESTING = False

MODEL_PATH = "models_pth/decoderSDF.pth"
LATENT_VECS_PATH = "models_pth/latent_vecs.pth"

MODEL_PATH_TEST = "models_pth/decoderSDF_TEST.pth"
LATENT_VECS_PATH_TEST = "models_pth/latent_vecs_TEST.pth"

# input_file = "../../data_processing/sdf_12_cars.h5"
input_file = "../sdf/sdf_input.h5"

latent_size = 16
num_epoch = 50000
batch_size = 10000

eta_decoder = 5e-3
eta_latent_space = 1e-2

# load file
h5f = h5py.File(input_file, 'r')

sdf_data = torch.Tensor(h5f["tensor"][()])

resolution = sdf_data.shape[1]
num_samples_per_scene = resolution * resolution * resolution
num_scenes = len(sdf_data)

assert(len(sdf_data.shape) == 5), "sdf data shoud have dimension: num_scenes x X_dim x Y_dim x Z_dim x 4 (sdf + r + g + b)"
assert(sdf_data.shape[1] == sdf_data.shape[2] and sdf_data.shape[2] == sdf_data.shape[3]),"resolution should be the same in every direction"

#fill tensors
idx = torch.arange(num_scenes).cuda()
xyz = torch.empty(num_samples_per_scene, 3,  dtype=torch.float).cuda()
sdf_gt = torch.empty(sdf_data.numel(), dtype=torch.float)
rgb_gt = torch.empty([sdf_data.numel(), 3], dtype=torch.uint8)

for x in range(resolution):
    for y in range(resolution):
        for z in range(resolution):
            xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

for id in range(num_scenes):
    if id%4 == 0:
        print(id)
    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                sdf_gt[id * num_samples_per_scene + x * resolution * resolution + y * resolution + z] = sdf_data[id,x,y,z,0]
                rgb_gt[id * num_samples_per_scene + x * resolution * resolution + y * resolution + z,:] = sdf_data[id,x,y,z,1:]

sdf_gt = sdf_gt.cuda()
sdf_gt = sdf_gt /resolution

rgb_gt = rgb_gt.cuda()

threshold_precision = 1
threshold_precision = threshold_precision/resolution


# initialize random latent code for every shape
lat_vecs = torch.nn.Embedding(num_scenes, latent_size).cuda()
torch.nn.init.normal_(
    lat_vecs.weight.data,
    0.0,
    1.0 / math.sqrt(latent_size),
)

# decoder
decoder = DecoderSDF(latent_size).cuda()


loss = torch.nn.MSELoss

#optimizer
optimizer = torch.optim.Adam(
    [
        {
            "params": decoder.parameters(),
            "lr": eta_decoder,
        },
        {
            "params": lat_vecs.parameters(),
            "lr": eta_latent_space,
        },
    ]
)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995)



####################### Training loop ##########################
decoder.train()

log_loss = []
log_loss_sdf = []
log_loss_rgb = []
log_loss_reg = []

for epoch in range(num_epoch):
    optimizer.zero_grad()

    #get random scene and samples
    batch_sample_idx = np.random.randint(num_samples_per_scene, size = batch_size)
    batch_scenes_idx = np.random.randint(num_scenes, size = batch_size)

    sdf_pred = decoder(lat_vecs(idx[batch_scenes_idx]), xyz[batch_sample_idx])

    # assign weight of 0 for easy samples that are well trained
    weight_sdf = ~((sdf_pred[:,0] > threshold_precision).squeeze() * (sdf_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx] > threshold_precision).squeeze()) \
        * ~((sdf_pred[:,0] < -threshold_precision).squeeze() * (sdf_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx] < -threshold_precision).squeeze())

    
    #L1 loss, only for hard samples
    loss_sdf = loss(reduction='none')(sdf_pred[:,0].squeeze(), sdf_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx])
    loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()

    # loss rgb
    lambda_rgb = 1/100
    
    rgb_gt_normalized = rgb_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx,:]/255
    loss_rgb = loss(reduction='none')(sdf_pred[:,1:], rgb_gt_normalized)
    loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean() * weight_sdf.numel()/weight_sdf.count_nonzero() * lambda_rgb
    

    #regularization loss
    lambda_reg_std = 1/1000
    lambda_reg_mean = 1/1000
    loss_reg_std = lambda_reg_std * abs(1.0 / math.sqrt(latent_size) - (lat_vecs.weight).std())
    loss_reg_mean = lambda_reg_mean * abs((lat_vecs.weight).mean())
    loss_reg = loss_reg_mean + loss_reg_std


    loss_pred = loss_sdf + loss_rgb + loss_reg_std + loss_reg_mean

    #log
    log_loss.append(loss_pred.detach().cpu())
    log_loss_sdf.append(loss_sdf.detach().cpu())
    log_loss_rgb.append(loss_rgb.detach().cpu())
    log_loss_reg.append(loss_reg.detach().cpu())

    #update weights
    loss_pred.backward()
    optimizer.step()
    scheduler.step()

    print("After {} epoch,  loss sdf: {:.5f}, loss rgb: {:.5f}, loss reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, min/max rgb: {:.2f}/{:.2f}, lr: {:f}, lat_vec mean/std: {:.2f}/{:.2f}".format(\
        epoch, torch.Tensor(log_loss_sdf[-10:]).mean(), torch.Tensor(log_loss_rgb[-10:]).mean(), torch.Tensor(log_loss_reg[-10:]).mean(), sdf_pred[:,0].min() * resolution, \
        sdf_pred[:,0].max() * resolution, sdf_pred[:,1:].min() * 255, sdf_pred[:,1:].max() * 255, optimizer.param_groups[0]['lr'], (lat_vecs.weight).std(), (lat_vecs.weight).mean()))



#save model
if (TESTING == True):
    torch.save(decoder, MODEL_PATH_TEST)
    torch.save(lat_vecs, LATENT_VECS_PATH_TEST)
else:
    torch.save(decoder, MODEL_PATH)
    torch.save(lat_vecs, LATENT_VECS_PATH)


print("final loss: {:f}".format(torch.Tensor(log_loss_sdf[-100:]).mean()))

#save logs plot
avrg_loss = []
avrg_loss_sdf = []
avrg_loss_rgb = []
for i in range(0,len(log_loss)):
    avrg_loss.append(torch.Tensor(log_loss[i-20:i]).mean())
    avrg_loss_sdf.append(torch.Tensor(log_loss_sdf[i-20:i]).mean())
    avrg_loss_rgb.append(torch.Tensor(log_loss_rgb[i-20:i]).mean())
    

import pickle
with open("logs/log.txt", "wb") as fp:
    pickle.dump(avrg_loss, fp)

from matplotlib import pyplot as plt
plt.figure()
plt.title("Total loss")
plt.semilogy(avrg_loss[:])
plt.savefig("../../data_processing/logs/log_total")
plt.figure()
plt.title("SDF loss")
plt.semilogy(avrg_loss_sdf[:])
plt.savefig("../../data_processing/logs/log_sdf")
plt.figure()
plt.title("RGB loss")
plt.semilogy(avrg_loss_rgb[:])
plt.savefig("../../data_processing/logs/log_rgb")

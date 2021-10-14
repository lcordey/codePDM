import h5py
import math
import numpy as np
import torch
import pickle
import argparse
import os
import IPython
import glob

from traitlets.traitlets import default

from networks import DecoderSDF
from marching_cubes_rgb import *

###### parameter #####

DEFAULT_SDF_DIR = '64'

MODEL_PATH = "models_pth/decoderSDF.pth"
LATENT_VECS_TARGET_PATH = "models_pth/latent_vecs_target.pth"

ALL_SDF_DIR_PATH = "../../image2sdf/sdf/"
ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"

latent_size = 16
num_epoch = 100000
batch_size = 10000

eta_decoder = 1e-3
eta_latent_space_mu = 5e-3
eta_latent_space_std = 1e-2
gammaLR = 0.99995


def load_sdf_data(input: str, annotations: dict) -> torch.tensor:

    sdf_dir_path = ALL_SDF_DIR_PATH + input + '/'

    if not os.path.exists(sdf_dir_path):
        print('Input directory does not exist.')
        exit(1)

    num_scenes = len(annotations.keys())
    sdf_data = None

    for scene_hash, scene_id in zip(annotations.keys(), range(num_scenes)):

        h5f = h5py.File(sdf_dir_path + scene_hash + '.h5', 'r')
        h5f_tensor = torch.tensor(h5f["tensor"][()], dtype = torch.float)

        if sdf_data is None:
            sdf_data = torch.empty([num_scenes, h5f_tensor.shape[0], h5f_tensor.shape[1], h5f_tensor.shape[2], h5f_tensor.shape[3]])

        sdf_data[scene_id, :,:,:,:] = h5f_tensor

    return sdf_data

def init_xyz(resolution):
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz

def init_gt(sdf_data, resolution, num_samples_per_scene, num_scenes):

    sdf_gt = np.reshape(sdf_data[:,:,:,:,0], [num_samples_per_scene * num_scenes])
    rgb_gt = np.reshape(sdf_data[:,:,:,:,1:], [num_samples_per_scene * num_scenes, 3])

    sdf_gt = sdf_gt.cuda()
    sdf_gt = sdf_gt /resolution

    rgb_gt = rgb_gt.cuda()

    return sdf_gt, rgb_gt

def init_lat_vecs(num_scenes, latent_size):

    #  initialize random latent code for every shape
    lat_vecs_mu = torch.nn.Embedding(num_scenes, latent_size).cuda()
    torch.nn.init.normal_(
        lat_vecs_mu.weight.data,
        0.0,
        1.0,
    )
    lat_vecs_log_std = torch.nn.Embedding(num_scenes, latent_size).cuda()
    torch.nn.init.normal_(
        lat_vecs_log_std.weight.data,
        0.0,
        0.0,
    )

    return lat_vecs_mu, lat_vecs_log_std

def init_opt_sched(decoder, lat_vecs_mu, lat_vecs_log_std):
    #optimizer
    optimizer = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": eta_decoder,
                "eps": 1e-8,
            },
            {
                "params": lat_vecs_mu.parameters(),
                "lr": eta_latent_space_mu,
                "eps": 1e-8,
            },
            {
                "params": lat_vecs_log_std.parameters(),
                "lr": eta_latent_space_std,
                "eps": 1e-8,
            },
        ]
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gammaLR)

    return optimizer, scheduler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('--input', type=str, help='The input directory containing HDF5 files.', default= DEFAULT_SDF_DIR)
    args = parser.parse_args()


    annotations_file = open(ANNOTATIONS_PATH, "rb")
    annotations = pickle.load(annotations_file)

    sdf_data = load_sdf_data(args.input, annotations)

    resolution = sdf_data.shape[1]
    threshold_precision = 1.0/resolution
    num_samples_per_scene = resolution * resolution * resolution
    num_scenes = len(sdf_data)

    #fill tensors
    idx = torch.arange(num_scenes).type(torch.LongTensor).cuda()
    xyz = init_xyz(resolution)

    sdf_gt, rgb_gt = init_gt(sdf_data, resolution, num_samples_per_scene, num_scenes)

    lat_vecs_mu, lat_vecs_log_std = init_lat_vecs(num_scenes, latent_size)


    # decoder
    decoder = DecoderSDF(latent_size).cuda()
    loss = torch.nn.MSELoss(reduction='none')
    optimizer, scheduler = init_opt_sched(decoder, lat_vecs_mu, lat_vecs_log_std)


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

        latent_code =  torch.empty_like(lat_vecs_log_std(idx[batch_scenes_idx])).normal_() * lat_vecs_log_std(idx[batch_scenes_idx]).exp() / 10 + lat_vecs_mu(idx[batch_scenes_idx])

        sdf_pred = decoder(latent_code, xyz[batch_sample_idx])


        # assign weight of 0 for easy samples that are well trained
        weight_sdf = ~((sdf_pred[:,0] > threshold_precision).squeeze() * (sdf_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx] > threshold_precision).squeeze()) \
            * ~((sdf_pred[:,0] < -threshold_precision).squeeze() * (sdf_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx] < -threshold_precision).squeeze())

        
        #L1 loss, only for hard samples
        loss_sdf = loss(sdf_pred[:,0].squeeze(), sdf_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx])
        loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()

        # loss rgb
        lambda_rgb = 1/100
        
        rgb_gt_normalized = rgb_gt[batch_scenes_idx * num_samples_per_scene + batch_sample_idx,:]/255
        loss_rgb = loss(sdf_pred[:,1:], rgb_gt_normalized)
        loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean() * weight_sdf.numel()/weight_sdf.count_nonzero() * lambda_rgb
        
        # regularization loss
        lambda_kl = 1/100
        loss_kl = (-0.5 * (1 + lat_vecs_log_std.weight - lat_vecs_mu.weight.pow(2) - lat_vecs_log_std.weight.exp())).mean()
        loss_kl = loss_kl * lambda_kl

        loss_pred = loss_sdf + loss_rgb + loss_kl

        #log
        log_loss.append(loss_pred.detach().cpu())
        log_loss_sdf.append(loss_sdf.detach().cpu())
        log_loss_rgb.append(loss_rgb.detach().cpu())
        log_loss_reg.append(loss_kl.detach().cpu())

        #update weights
        loss_pred.backward()
        optimizer.step()
        scheduler.step()

        print("After {} epoch,  loss sdf: {:.5f}, loss rgb: {:.5f}, loss reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, min/max rgb: {:.2f}/{:.2f}, lr: {:f}, lat_vec std/mu: {:.2f}/{:.2f}".format(\
            epoch, torch.Tensor(log_loss_sdf[-10:]).mean(), torch.Tensor(log_loss_rgb[-10:]).mean(), torch.Tensor(log_loss_reg[-10:]).mean(), sdf_pred[:,0].min() * resolution, \
            sdf_pred[:,0].max() * resolution, sdf_pred[:,1:].min() * 255, sdf_pred[:,1:].max() * 255, optimizer.param_groups[0]['lr'], (lat_vecs_log_std.weight.exp()).mean(), (lat_vecs_mu.weight).abs().mean()))

    #save model
    torch.save(decoder, MODEL_PATH)
    torch.save(lat_vecs_mu(idx).detach(), LATENT_VECS_TARGET_PATH)



    print("final loss sdf: {:f}".format(torch.Tensor(log_loss_sdf[-100:]).mean()))
    print("final loss rgb: {:f}".format(torch.Tensor(log_loss_rgb[-100:]).mean()))


    #save sdf results
    sdf_output = np.empty([num_scenes , resolution, resolution, resolution, 4], dtype = np.float16)

    decoder.eval()
    for i in range(num_scenes):
        
        # free variable for memory space
        try:
            del sdf_pred
        except:
            print("sdf_pred wasn't defined")

        sdf_result = np.empty([resolution, resolution, resolution, 4], dtype = np.float16)
        for x in range(resolution):

            sdf_pred = decoder(lat_vecs_mu(idx[i].repeat(resolution * resolution)),xyz[x * resolution * resolution: (x+1) * resolution * resolution])

            sdf_pred[:,0] = sdf_pred[:,0] * resolution
            sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
            sdf_pred[:,1:] = sdf_pred[:,1:] * 255
            
            sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:].detach().cpu(), [resolution, resolution, 4])

        sdf_output[i] = sdf_result

        print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
        if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
            vertices, faces = marching_cubes(sdf_result[:,:,:,0])
            colors_v = exctract_colors_v(vertices, sdf_result)
            colors_f = exctract_colors_f(colors_v, faces)
            off_file = '../../image2sdf/output_decoder/%d.off' % i
            write_off(off_file, vertices, faces, colors_f)
            print('Wrote %s.' % off_file)
        else:
            print("surface level: 0, should be comprise in between the minimum and maximum value")


    #save sdf
    with h5py.File('../../image2sdf/sdf/sdf_output_decoder.h5', 'w') as f:
        dset = f.create_dataset("tensor", data = sdf_output)


    #save logs plot
    avrg_loss = []
    avrg_loss_sdf = []
    avrg_loss_rgb = []
    for i in range(0,len(log_loss)):
        avrg_loss.append(torch.Tensor(log_loss[i-20:i]).mean())
        avrg_loss_sdf.append(torch.Tensor(log_loss_sdf[i-20:i]).mean())
        avrg_loss_rgb.append(torch.Tensor(log_loss_rgb[i-20:i]).mean())
        

    from matplotlib import pyplot as plt
    plt.figure()
    plt.title("Total loss")
    plt.semilogy(avrg_loss[:])
    plt.savefig("../../image2sdf/logs/log_total")
    plt.figure()
    plt.title("SDF loss")
    plt.semilogy(avrg_loss_sdf[:])
    plt.savefig("../../image2sdf/logs/log_sdf")
    plt.figure()
    plt.title("RGB loss")
    plt.semilogy(avrg_loss_rgb[:])
    plt.savefig("../../image2sdf/logs/log_rgb")

    with open("../../image2sdf/logs/log.txt", "wb") as fp:
        pickle.dump(avrg_loss, fp)

import h5py
import math
import numpy as np
from numpy.lib.financial import ipmt
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


DECODER_PATH = "models_pth/decoderSDF_VAE.pth"
ENCODER_PATH = "models_pth/encoderSDF_VAE.pth"

ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"
ALL_SDF_DIR_PATH = "../../image2sdf/sdf/"

DEFAULT_SDF_DIR = '64'


num_epoch = 200000
batch_size_scene = 5
batch_size_sample = 3000
latent_size = 16

eta_encoder = 5e-4
eta_decoder = 1e-3
# eta_decoder = 5e-4
# gammaLR = 0.99995
gammaLR = 0.99999

ratio_image_used = 0.5

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def load_encoder_input(annotations: dict, num_scene: int, num_image_per_scene: int):
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

    print("images loaded")
    return input_images, input_locations

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

    print("sdf data loaded")
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

def init_opt_sched(encoder, decoder):
    #optimizer
    optimizer = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": eta_decoder,
                "eps": 1e-8,
            },
            {
                "params": encoder.parameters(),
                "lr": eta_encoder,
                "eps": 1e-8,
            },
        ]
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gammaLR)

    return optimizer, scheduler

def evaluate_on_validation_datas(encoder, decoder, num_scene, num_validation_image_per_scene, num_samples_per_scene, sdf_gt, rgb_gt, validation_input_im, validation_input_loc):

    encoder.eval()
    decoder.eval()

    sum_loss_sdf = 0
    sum_loss_rgb = 0

    for scene_id in range(num_scene):
        for image_id in range(num_validation_image_per_scene):

            latent_code_validation = encoder(validation_input_im[scene_id, image_id, :, :, :].unsqueeze(0), validation_input_loc[scene_id,image_id, :].unsqueeze(0))[:,:latent_size].detach()
            sdf_pred_validation = decoder(latent_code_validation.repeat(num_samples_per_scene,1), xyz).detach()

            # assign weight of 0 for easy samples that are well trained
            weight_sdf_validation = ~((sdf_pred_validation[:,0] > threshold_precision).squeeze() * (sdf_gt[scene_id * num_samples_per_scene : (scene_id + 1) * num_samples_per_scene] > threshold_precision).squeeze()) \
                * ~((sdf_pred_validation[:,0] < -threshold_precision).squeeze() * (sdf_gt[scene_id * num_samples_per_scene : (scene_id + 1) * num_samples_per_scene] < -threshold_precision).squeeze())

            
            #L1 loss, only for hard samples
            loss_sdf_validation = loss(sdf_pred_validation[:,0].squeeze(), sdf_gt[scene_id * num_samples_per_scene : (scene_id + 1) * num_samples_per_scene])
            loss_sdf_validation = (loss_sdf_validation * weight_sdf_validation).mean() * weight_sdf_validation.numel()/weight_sdf_validation.count_nonzero()

            # loss rgb
            lambda_rgb = 1/100
            
            rgb_gt_normalized = rgb_gt[scene_id * num_samples_per_scene : (scene_id + 1) * num_samples_per_scene,:]/255
            loss_rgb_validation = loss(sdf_pred_validation[:,1:], rgb_gt_normalized)
            loss_rgb_validation = ((loss_rgb_validation[:,0] * weight_sdf_validation) + (loss_rgb_validation[:,1] * weight_sdf_validation) + (loss_rgb_validation[:,2] * weight_sdf_validation)).mean() * weight_sdf_validation.numel()/weight_sdf_validation.count_nonzero() * lambda_rgb
            
            sum_loss_sdf = sum_loss_sdf + loss_sdf_validation
            sum_loss_rgb = sum_loss_rgb + loss_rgb_validation

    sum_loss_sdf = sum_loss_sdf/(num_scene * num_validation_image_per_scene)
    sum_loss_rgb = sum_loss_rgb/(num_scene * num_validation_image_per_scene)

    print("****************************** VALIDATION ******************************")
    print("loss sdf: {:.5f}, loss rgb: {:.5f}, min/max sdf: {:.2f}/{:.2f}, min/max rgb: {:.2f}/{:.2f}".format(\
                sum_loss_sdf, sum_loss_rgb, sdf_pred_validation[:,0].min() * resolution, \
                sdf_pred_validation[:,0].max() * resolution, sdf_pred_validation[:,1:].min() * 255, sdf_pred_validation[:,1:].max() * 255))

    print("\n \n")

    encoder.train()
    decoder.train()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('--input', type=str, help='The input directory containing HDF5 files.', default= DEFAULT_SDF_DIR)
    args = parser.parse_args()

    annotations_file = open(ANNOTATIONS_PATH, "rb")
    annotations = pickle.load(annotations_file)

    num_image_per_scene = len(annotations[next(iter(annotations.keys()))])
    # num_image_per_scene = 5
    num_scene = len(annotations.keys())


    # encoder input
    input_images, input_locations = load_encoder_input(annotations, num_scene, num_image_per_scene)

    ratio_training_validation = 0.8

    num_training_image_per_scene = (np.int)(np.round(num_image_per_scene * ratio_image_used * ratio_training_validation))
    num_validation_image_per_scene = (np.int)(np.round(num_image_per_scene * ratio_image_used)) - num_training_image_per_scene

    rand_idx = np.arange(num_image_per_scene)
    np.random.shuffle(rand_idx)
    train_images_idx = rand_idx[num_training_image_per_scene]
    validation_images_idx = rand_idx[num_training_image_per_scene : num_training_image_per_scene + num_validation_image_per_scene]


    # num_training_image_per_scene = (np.int)(np.round(num_image_per_scene * ratio_training_validation))
    # num_validation_image_per_scene = num_image_per_scene - num_training_image_per_scene

    # train_images_idx = np.arange(num_training_image_per_scene)
    # validation_images_idx = np.arange(num_training_image_per_scene, num_image_per_scene)


    train_input_im = torch.tensor(input_images[:,train_images_idx,:,:,:], dtype = torch.float).cuda()
    validation_input_im = torch.tensor(input_images[:,validation_images_idx,:,:,:], dtype = torch.float).cuda()
    train_input_loc = torch.tensor(input_locations[:,train_images_idx,:], dtype = torch.float).cuda()
    validation_input_loc = torch.tensor(input_locations[:,validation_images_idx,:], dtype = torch.float).cuda()

    print(torch.cuda.memory_allocated(0)/torch.cuda.memory_reserved(0))
    
    # decoder target
    sdf_data = load_sdf_data(args.input, annotations)
    assert(num_scene == len(sdf_data)), "sdf folder should correspond to annotations input file"

    print(torch.cuda.memory_allocated(0)/torch.cuda.memory_reserved(0))

    resolution = sdf_data.shape[1]
    threshold_precision = 1.0/resolution
    num_samples_per_scene = resolution * resolution * resolution

    xyz = init_xyz(resolution)
    sdf_gt, rgb_gt = init_gt(sdf_data, resolution, num_samples_per_scene, num_scene)


    # encoder
    encoder = EncoderSDF(latent_size, vae = True).cuda()
    encoder.apply(init_weights)
    decoder = DecoderSDF(latent_size).cuda()
    decoder.apply(init_weights)

    optimizer, scheduler = init_opt_sched(encoder, decoder)

    loss = torch.nn.MSELoss(reduction='none')

    print("start taining")

    log_loss = []
    log_loss_sdf = []
    log_loss_rgb = []
    log_loss_reg = []

    encoder.train()
    decoder.train()
    for epoch in range(num_epoch):

        optimizer.zero_grad()

        # random batch
        batch_scene_idx = np.random.randint(num_scene, size = batch_size_scene)
        batch_image_idx = np.random.randint(num_training_image_per_scene, size = batch_size_scene)

        latent_code_mu_std = encoder(train_input_im[batch_scene_idx, batch_image_idx, :, :, :], train_input_loc[batch_scene_idx,batch_image_idx, :])

        latent_code_mu = latent_code_mu_std[:, :latent_size]
        latent_code_std = latent_code_mu_std[:, latent_size:]

        latent_code = torch.empty_like(latent_code_std).normal_() * latent_code_std.exp() / 10 + latent_code_mu

        latent_code = latent_code.repeat_interleave(batch_size_sample, dim=0)
        batch_scene_idx = np.repeat(batch_scene_idx, batch_size_sample)
        batch_sample_idx = np.tile(np.random.randint(num_samples_per_scene, size = batch_size_sample), batch_size_scene)

        sdf_pred = decoder(latent_code, xyz[batch_sample_idx])

        # assign weight of 0 for easy samples that are well trained
        weight_sdf = ~((sdf_pred[:,0] > threshold_precision).squeeze() * (sdf_gt[batch_scene_idx * num_samples_per_scene + batch_sample_idx] > threshold_precision).squeeze()) \
            * ~((sdf_pred[:,0] < -threshold_precision).squeeze() * (sdf_gt[batch_scene_idx * num_samples_per_scene + batch_sample_idx] < -threshold_precision).squeeze())

        
        #L1 loss, only for hard samples
        loss_sdf = loss(sdf_pred[:,0].squeeze(), sdf_gt[batch_scene_idx * num_samples_per_scene + batch_sample_idx])
        loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()

        # loss rgb
        lambda_rgb = 1/100
        
        rgb_gt_normalized = rgb_gt[batch_scene_idx * num_samples_per_scene + batch_sample_idx,:]/255
        loss_rgb = loss(sdf_pred[:,1:], rgb_gt_normalized)
        loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean() * weight_sdf.numel()/weight_sdf.count_nonzero() * lambda_rgb
        
        # regularization loss
        lambda_kl = 1/100
        loss_kl = (-0.5 * (1 + latent_code_std - latent_code_mu.pow(2) - latent_code_std.exp())).mean()
        loss_kl = loss_kl * lambda_kl

        loss_pred = loss_sdf + loss_rgb + loss_kl
        # loss_pred = loss_sdf + loss_rgb


        log_loss.append(loss_pred.detach().cpu())
        log_loss_sdf.append(loss_sdf.detach().cpu())
        log_loss_rgb.append(loss_rgb.detach().cpu())
        log_loss_reg.append(loss_kl.detach().cpu())

        #update weights
        loss_pred.backward()
        optimizer.step()
        scheduler.step()

        # print("After {} epoch,  loss sdf: {:.5f}, loss rgb: {:.5f}, loss reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, min/max rgb: {:.2f}/{:.2f}, lr: {:f}".format(\
        #     epoch, torch.Tensor(log_loss_sdf[-10:]).mean(), torch.Tensor(log_loss_rgb[-10:]).mean(), torch.Tensor(log_loss_reg[-10:]).mean(), sdf_pred[:,0].min() * resolution, \
        #     sdf_pred[:,0].max() * resolution, sdf_pred[:,1:].min() * 255, sdf_pred[:,1:].max() * 255, optimizer.param_groups[0]['lr']))


        print("After {} epoch,  loss sdf: {:.5f}, loss rgb: {:.5f}, loss reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, min/max rgb: {:.2f}/{:.2f}, lr: {:f}, lat_vec std/mu: {:.2f}/{:.2f}".format(\
            epoch, torch.Tensor(log_loss_sdf[-10:]).mean(), torch.Tensor(log_loss_rgb[-10:]).mean(), torch.Tensor(log_loss_reg[-10:]).mean(), sdf_pred[:,0].min() * resolution, \
            sdf_pred[:,0].max() * resolution, sdf_pred[:,1:].min() * 255, sdf_pred[:,1:].max() * 255, optimizer.param_groups[0]['lr'], (latent_code_std.exp()).mean(), (latent_code_mu).abs().mean()))
            
        if epoch %500 == 0:
            evaluate_on_validation_datas(encoder, decoder, num_scene, num_validation_image_per_scene, num_samples_per_scene, sdf_gt, rgb_gt, validation_input_im, validation_input_loc)


    evaluate_on_validation_datas(encoder, decoder, num_scene, num_validation_image_per_scene, num_samples_per_scene, sdf_gt, rgb_gt, validation_input_im, validation_input_loc)

    torch.save(encoder, ENCODER_PATH)
    torch.save(decoder, DECODER_PATH)

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

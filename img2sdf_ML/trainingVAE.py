import h5py
import math
import numpy as np
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


DECODER_PATH = "models_pth/decoderSDF.pth"
ENCODER_PATH = "models_pth/encoderSDF.pth"

ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"
ALL_SDF_DIR_PATH = "../../image2sdf/sdf/"

DEFAULT_SDF_DIR = '64'


num_epoch = 50000
batch_size = 25
latent_size = 16

eta_encoder = 5e-4
eta_decoder = 1e-3
gammaLR = 0.99995

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('--input', type=str, help='The input directory containing HDF5 files.', default= DEFAULT_SDF_DIR)
    args = parser.parse_args()

    annotations_file = open(ANNOTATIONS_PATH, "rb")
    annotations = pickle.load(annotations_file)

    # num_image_per_scene = len(annotations[next(iter(annotations.keys()))])
    num_image_per_scene = 2
    num_scene = len(annotations.keys())


    # encoder input
    input_images, input_locations = load_encoder_input(annotations, num_scene, num_image_per_scene)

    ratio_training_validation = 0.8
    num_training_image_per_scene = (np.int)(np.round(num_image_per_scene * ratio_training_validation))
    num_validation_image_per_scene = num_image_per_scene - num_training_image_per_scene

    train_images_idx = np.arange(num_training_image_per_scene)
    validation_images_idx = np.arange(num_training_image_per_scene, num_image_per_scene)

    train_input_im = torch.tensor(input_images[:,train_images_idx,:,:,:], dtype = torch.float).cuda()
    validation_input_im = torch.tensor(input_images[:,validation_images_idx,:,:,:], dtype = torch.float).cuda()
    train_input_loc = torch.tensor(input_locations[:,train_images_idx,:], dtype = torch.float).cuda()
    validation_input_loc = torch.tensor(input_locations[:,validation_images_idx,:], dtype = torch.float).cuda()

    
    # decoder target
    sdf_data = load_sdf_data(args.input, annotations)
    assert(num_scene == len(sdf_data)), "sdf folder should correspond to annotations input file"

    resolution = sdf_data.shape[1]
    threshold_precision = 1.0/resolution
    num_samples_per_scene = resolution * resolution * resolution

    xyz = init_xyz(resolution)
    sdf_gt, rgb_gt = init_gt(sdf_data, resolution, num_samples_per_scene, num_scene)


    # encoder
    encoder = EncoderSDF(latent_size).cuda()
    encoder.apply(init_weights)
    decoder = DecoderSDF(latent_size).cuda()
    decoder.apply(init_weights)

    optimizer, scheduler = init_opt_sched(encoder, decoder)

    loss = torch.nn.MSELoss(reduction='none')

    print("done")




""" 
DataLoader for the encoder and decoder training
"""

import torch
import pickle
import imageio
import cv2
import numpy as np
import time
from utils import *


class DatasetDecoder(torch.utils.data.Dataset):
    "This dataset is used to train the Decoder, it returns a random set of sample from random models with their ground truth sdf and rgb values"

    def __init__(self, list_hash, dict_gt_data, num_samples_per_model, dict_model_hash_2_idx):
        """ 
        list_hash: list of strings, each the hash of a model to load
        dict_gt_data: Dictionnary that contains the ground truth value given a model and an sample ID
        num_samples_per_model: number of training samples per model
        dict_model_hash_2_idx: Dictionnary that returns the ID corresponding to a model hash
        """
        
        self.list_hash = list_hash
        self.dict_gt_data = dict_gt_data
        self.num_samples_per_model = num_samples_per_model
        self.dict_model_hash_2_idx = dict_model_hash_2_idx

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_hash) * self.num_samples_per_model

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        xyz_idx = index%self.num_samples_per_model
        model_hash = self.list_hash[(int)((index - xyz_idx)/self.num_samples_per_model)]
        # get model idx
        model_idx = self.dict_model_hash_2_idx[model_hash]

        # get sdf and rgb values
        sdf_gt = self.dict_gt_data["sdf"][model_hash][xyz_idx]
        rgb_gt = self.dict_gt_data["rgb"][model_hash][xyz_idx]

        return model_idx, sdf_gt, rgb_gt, xyz_idx

class DatasetGrid(torch.utils.data.Dataset):
    "This dataset is used to train the Encoder, it returns a 3D tensor generated with an image and a bounding box"

    def __init__(self, list_hash, annotations, num_images_per_model, param_image, param_network, image_path):
        """ 
        list_hash: list of strings, each the hash of a model to load
        annotations: Dictionnary annotations that contains the 3D position of the bounding box,
                     as well as the intrinsics and extrinsics matrices
        num_samples_per_model: number of training images per model
        param_image: contains the specs of the images
        param_network: contains the specs of the networks, can be modified in the "config/param.yaml" file
        image_path: path of the folder containing the images
        """

        self.list_hash = list_hash
        self.annotations = annotations
        self.num_images_per_model = num_images_per_model
        self.width_image = param_image["width"]
        self.height_image = param_image["height"]
        self.num_slices = param_network["num_slices"]
        self.width_network = param_network["width"]
        self.height_network = param_network["height"]
        self.image_path = image_path
        self.matrix_world_to_camera = annotations["matrix_world_to_camera"]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_hash) * self.num_images_per_model

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        image_id = index%self.num_images_per_model
        model_hash = self.list_hash[(int)((index - image_id) / self.num_images_per_model)]

        # Load data and get label
        image_pth = self.image_path + model_hash + '/' + str(image_id) + '.png'
        input_im = imageio.imread(image_pth)

        # loc_2d = self.annotations[scene_id][rand_image_id]['2d'].copy()
        loc_3d = self.annotations[model_hash][image_id]['3d'].copy()
        frame = self.annotations[model_hash][image_id]['frame'].copy()

        # interpolate slices vertex coordinates
        loc_slice_3d = np.empty([self.num_slices,4,3])
        for i in range(self.num_slices):
            loc_slice_3d[i,0,:] = loc_3d[0,:] * (1-i/(self.num_slices-1)) + loc_3d[4,:] * i/(self.num_slices-1)
            loc_slice_3d[i,1,:] = loc_3d[1,:] * (1-i/(self.num_slices-1)) + loc_3d[5,:] * i/(self.num_slices-1)
            loc_slice_3d[i,2,:] = loc_3d[2,:] * (1-i/(self.num_slices-1)) + loc_3d[6,:] * i/(self.num_slices-1)
            loc_slice_3d[i,3,:] = loc_3d[3,:] * (1-i/(self.num_slices-1)) + loc_3d[7,:] * i/(self.num_slices-1)

        # convert to image plane coordinate
        loc_slice_2d = np.empty_like(loc_slice_3d)
        for i in range(self.num_slices):
            for j in range(4):
                    loc_slice_2d[i,j,:] = convert_w2c(self.matrix_world_to_camera, frame, loc_slice_3d[i,j,:]) 

        ###### y coordinate is inverted + rescaling #####
        loc_slice_2d[:,:,1] = 1 - loc_slice_2d[:,:,1]
        loc_slice_2d[:,:,0] = loc_slice_2d[:,:,0] * self.width_image
        loc_slice_2d[:,:,1] = loc_slice_2d[:,:,1] * self.height_image

        # grid to give as input to the network
        input_grid = np.empty([self.num_slices, self.width_network, self.height_network, 3])

        # fill grid by slices
        for i in range(self.num_slices):
            src = loc_slice_2d[i,:,:2].copy()
            dst = np.array([[0, self.height_network], [self.width_network, self.height_network], [self.width_network, 0], [0,0]])
            h, mask = cv2.findHomography(src, dst)
            slice = cv2.warpPerspective(input_im, h, (self.width_network,self.height_network))
            input_grid[i,:,:,:] = slice

        # rearange, normalize and convert to tensor
        input_grid = np.transpose(input_grid, [3,0,1,2])
        input_grid = input_grid/255 - 0.5
        input_grid = torch.tensor(input_grid, dtype = torch.float)

        return input_grid, model_hash


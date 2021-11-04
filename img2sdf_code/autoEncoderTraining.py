import numpy as np
import torch
import torch.nn as nn
import json
import pickle
import time
import h5py
import glob
import os

from networks import EncoderGrid

import IPython


ENCODER_PATH = "models_and_codes/encoderGrid.pth"
DECODER_PATH = "models_and_codes/decoder.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
PARAM_FILE = "config/param.json"
VEHICLE_VALIDATION_PATH = "config/vehicle_validation.txt"
ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
LOGS_PATH = "../../image2sdf/logs/log.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"
MATRIX_PATH = "../../image2sdf/input_images/matrix_w2c.pkl"
SDF_DIR = "../../image2sdf/sdf/"




def init_xyz(resolution):
    """ fill 3d grid representing 3d location to give as input to the decoder """
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz



if __name__ == '__main__':
    print("Loading parameters...")

    # load parameters
    param_all = json.load(open(PARAM_FILE))
    param = param_all["decoder"]
    resolution = param_all["resolution_used_for_training"]

    threshold_precision = 1.0/resolution
    num_samples_per_model = resolution * resolution * resolution

    # load annotations and validation hash list
    annotations = pickle.load(open(ANNOTATIONS_PATH, 'rb'))
    with open(VEHICLE_VALIDATION_PATH) as f:
        list_hash_validation = f.read().splitlines()
    list_hash_validation = list(list_hash_validation)

    # get models' hashs
    list_model_hash = []
    for val in glob.glob(SDF_DIR + "*.h5"):
        model_hash = os.path.basename(val).split('.')[0]
        if model_hash not in list_hash_validation: # check that this model is not used for validation
            if model_hash in annotations.keys(): # check that we have annotation for this model
                list_model_hash.append(model_hash)

    num_model = len(list_model_hash)

    # load every models
    print("Loading models...")
    dict_gt_data = dict()
    dict_gt_data["sdf"] = dict()
    dict_gt_data["rgb"] = dict()

    for model_hash, i in zip(list_model_hash, range(num_model)):
        if i%25 == 0:
            print(f"loading models: {i}/{num_model}")

        # load sdf tensor
        h5f = h5py.File(SDF_DIR + model_hash + '.h5', 'r')
        h5f_tensor = torch.tensor(h5f["tensor"][()], dtype = torch.float)

        # split sdf and rgb then reshape
        sdf_gt = np.reshape(h5f_tensor[:,:,:,0], [num_samples_per_model])
        rgb_gt = np.reshape(h5f_tensor[:,:,:,1:], [num_samples_per_model , 3])

        # normalize
        sdf_gt = sdf_gt / resolution
        rgb_gt = rgb_gt / 255

        # store in dict
        dict_gt_data["sdf"][model_hash] = sdf_gt
        dict_gt_data["rgb"][model_hash] = rgb_gt






    

    # fill a xyz grid to give as input to the decoder 
    xyz = init_xyz(resolution)
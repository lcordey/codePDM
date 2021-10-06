 import h5py
import math
import numpy as np
import torch
import pickle
import argparse
import os

from networks import DecoderSDF
from marching_cubes_rgb import *

###### parameter #####

TESTING = False

MODEL_PATH = "models_pth/decoderSDF.pth"
LATENT_VECS_PATH = "models_pth/latent_vecs.pth"

MODEL_PATH_TEST = "models_pth/decoderSDF_TEST.pth"
LATENT_VECS_PATH_TEST = "models_pth/latent_vecs_TEST.pth"

# input_file = "../../data_processing/sdf/sdf.h5"
input_dir = "../../data_processing/sdf/"

latent_size = 16
num_epoch = 5000
batch_size = 10000

eta_decoder = 1e-3
eta_latent_space = 1e-2
gammaLR = 0.99995


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('input', type=str, help='The input HDF5 file.')
    # parser.add_argument('output', type=str, help='Output directory for OFF files.')

    args = parser.parse_args()

    path_input = input_dir + args.input

    if not os.path.exists(path_input):
        print('Input file does not exist.')
        exit(1)

    # load file
    h5f = h5py.File(path_input, 'r')

    # sdf_data = torch.tensor(h5f["tensor"][()], dtype = torch.half)
    sdf_data = torch.tensor(h5f["tensor"][()], dtype = torch.float)

import h5py
import math
import numpy as np
import torch
import pickle
import argparse
import os
import IPython
import glob

from networks import DecoderSDF, EncoderSDF
from marching_cubes_rgb import *


DEFAULT_SDF_DIR = '64'

DECODER_PATH = "models_pth/decoderSDF.pth"
ENCODER_PATH = "models_pth/encoderSDF.pth"

ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"
ALL_SDF_DIR_PATH = "../../image2sdf/sdf/"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('--input', type=str, help='The input directory containing HDF5 files.', default= DEFAULT_SDF_DIR)
    args = parser.parse_args()


    annotations_file = open(ANNOTATIONS_PATH, "rb")
    annotations = pickle.load(annotations_file)

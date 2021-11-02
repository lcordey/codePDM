import numpy as np
import torch
import json
import pickle

from networks import *
from dataLoader import DatasetGrid
from marching_cubes_rgb import *



DECODER_PATH = "models_and_codes/decoder.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
ENCODER_PATH = "models_and_codes/encoderGrid.pth"
PARAM_FILE = "config/param.json"
ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"
MATRIX_PATH = "../../image2sdf/input_images/matrix_w2c.pkl"


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def cosine_distance(a,b):
    return a.dot(b)/(a.norm() * b.norm())


def init_opt_sched(encoder, param):
    """ initialize optimizer and scheduler"""

    optimizer = torch.optim.Adam(
        [
            {
                "params": encoder.parameters(),
                "lr": param["eta_encoder"],
                "eps": 1e-8,
            },
        ]
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=param["gammaLR"])

    return optimizer, scheduler


if __name__ == '__main__':
    print("Loading parameters...")

    # load parameters
    param_all = json.load(open(PARAM_FILE))
    param = param_all["encoder"]

    # Load decoder
    decoder = torch.load(DECODER_PATH).cuda()

    # load codes and annotations
    dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))
    annotations = pickle.load(open(ANNOTATIONS_PATH, 'rb'))


    # Only consider model which appear in both annotation and code
    list_hash = []
    for hash in dict_hash_2_code.keys():
        if hash in annotations.keys():
            list_hash.append(hash)

    num_model = len(list_hash)

    num_image_per_model = len(annotations[list_hash[0]])
    latent_size = dict_hash_2_code[list_hash[0]].shape[0]

    # Init training dataset
    training_set = DatasetGrid(list_hash, annotations, num_image_per_model, param["image"], param["network"], IMAGES_PATH, MATRIX_PATH)
    training_generator= torch.utils.data.DataLoader(training_set, **param["dataLoader"])

    IPython.embed()

    # Init Encoder
    encoder = EncoderGrid(latent_size).cuda()
    encoder.apply(init_weights)

    # initialize optimizer and scheduler
    optimizer, scheduler = init_opt_sched(encoder, param["optimizer"])
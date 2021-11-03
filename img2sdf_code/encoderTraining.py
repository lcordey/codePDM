import numpy as np
import torch
import torch.nn as nn
import json
import pickle
import time

from networks import EncoderGrid
from dataLoader import DatasetGrid

import IPython


ENCODER_PATH = "models_and_codes/encoderGrid.pth"
DECODER_PATH = "models_and_codes/decoder.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
PARAM_FILE = "config/param.json"
ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
LOGS_PATH = "../../image2sdf/logs/log.pkl"
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

def compute_time_left(time_start, samples_count, num_model, num_images_per_model, epoch, num_epoch):
    """ Compute time left until the end of training """
    time_passed = time.time() - time_start
    num_samples_seen = epoch * num_model * num_images_per_model + samples_count
    time_per_sample = time_passed/num_samples_seen
    estimate_total_time = time_per_sample * num_epoch * num_model * num_images_per_model
    estimate_time_left = estimate_total_time - time_passed

    return estimate_time_left



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
    for hash in annotations.keys():
        if hash in dict_hash_2_code.keys():
            list_hash.append(hash)

    num_model = len(list_hash)
    num_images_per_model = len(annotations[list_hash[0]])
###################################### change ######################################
    num_images_per_model = 300
###################################### change ######################################
    latent_size = dict_hash_2_code[list_hash[0]].shape[0]

    # Init training dataset
    training_set = DatasetGrid(list_hash, annotations, num_images_per_model, param["image"], param["network"], IMAGES_PATH, MATRIX_PATH)
    training_generator= torch.utils.data.DataLoader(training_set, **param["dataLoader"])

    # Init Encoder
    encoder = EncoderGrid(latent_size).cuda()
    encoder.apply(init_weights)

    # initialize optimizer and scheduler
    optimizer, scheduler = init_opt_sched(encoder, param["optimizer"])
    loss = torch.nn.MSELoss()



    # logs
    logs = dict()
    logs["training"] = []



    encoder.train()
    print("Start trainging...")

    time_start = time.time()
    
    for epoch in range(param["num_epoch"]):
        samples_count = 0
        for batch_images, batch_model_hash in training_generator:
            optimizer.zero_grad()
            batch_size = len(batch_images)

            # transfer to gpu
            batch_images = batch_images.cuda()

            # get target code
            target_code = torch.empty([batch_size, latent_size]).cuda()
            for model_hash, i in zip(batch_model_hash, range(batch_size)):
                target_code[i] = dict_hash_2_code[model_hash]


            predicted_code = encoder(batch_images)

            # compute loss
            loss_training = loss(predicted_code, target_code)
            logs["training"].append(loss_training.detach().cpu())

            #update weights
            loss_training.backward()
            optimizer.step()

            # compute time left
            samples_count += batch_size
            time_left = compute_time_left(time_start, samples_count, num_model, num_images_per_model, epoch, param["num_epoch"])

            # print everyl X model seen
            if samples_count%(10 * batch_size) == 0:
                print("epoch: {}/{:.2f}%, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}, time left: {} min".format(\
                    epoch, 100 * samples_count / (num_model * num_images_per_model), loss_training, \
                    abs(predicted_code - target_code).mean(), abs(predicted_code).mean(), abs(target_code).mean(),\
                    optimizer.param_groups[0]['lr'],  (int)(time_left/60) ))

        scheduler.step()

    print(f"Training finish in {(int)((time.time() - time_start) / 60)} min")


    ###### Saving eNcoder ######
    # save encoder
    torch.save(encoder, ENCODER_PATH)

    # save logs
    with open(LOGS_PATH, "wb") as fp:
        pickle.dump(logs, fp)
        


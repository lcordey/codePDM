import numpy as np
import torch
import torch.nn as nn
import json
import pickle
import time
import h5py
import glob
import os

from dataLoader import DatasetVAE
from networks import Decoder, EncoderGrid

import IPython

with open("/etc/hostname", "r") as f:
    identity = f.read()
    if identity == "loic-laptop\n":
        exit()

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



def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def init_xyz(resolution):
    """ fill 3d grid representing 3d location to give as input to the decoder """
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz


def init_opt_sched(encoder, decoder, param):
    """ initialize optimizer and scheduler"""

    optimizer = torch.optim.Adam(
        [
            {
                "params": encoder.parameters(),
                "lr": param["eta_encoder"],
            },
            {
                "params": decoder.parameters(),
                "lr": param["eta_decoder"],
            }
        ]
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=param["gammaLR"])

    return optimizer, scheduler


# def compute_loss(pred_sdf, pred_rgb, sdf_gt, rgb_gt, lat_code_mu, lat_code_log_std, threshold_precision, param):
def compute_loss(pred_sdf, pred_rgb, sdf_gt, rgb_gt, predicted_code, threshold_precision, param):
    """ compute sdf, rgb and regression loss """

    loss = torch.nn.MSELoss(reduction='none')

    # assign weight of 0 for easy samples that are well trained
    weight_sdf = ~((pred_sdf > threshold_precision).squeeze() * (sdf_gt > threshold_precision).squeeze()) \
        * ~((pred_sdf < -threshold_precision).squeeze() * (sdf_gt < -threshold_precision).squeeze())

    #L2 loss, only for hard samples
    loss_sdf = loss(pred_sdf.squeeze(), sdf_gt)
    loss_sdf = (loss_sdf * weight_sdf).sum()/weight_sdf.count_nonzero()
    loss_sdf *= param["lambda_sdf"]

    # loss rgb
    loss_rgb = loss(pred_rgb, rgb_gt)
    loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).sum()/weight_sdf.count_nonzero()
    loss_rgb *= param["lambda_rgb"]
    
    # regularization loss
    # loss_kl = (-0.5 * (1 + lat_code_log_std.weight - lat_code_mu.weight.pow(2) - lat_code_log_std.weight.exp())).mean()
    loss_kl = (-0.5 * (1 + 0 - predicted_code.pow(2) - 1)).mean()
    loss_kl *= param["lambda_kl"]

    return loss_sdf, loss_rgb, loss_kl


def compute_time_left(time_start, samples_count, num_model, num_samples_per_model, num_images_per_model, epoch, num_epoch):
    """ Compute time left until the end of training """
    time_passed = time.time() - time_start
    num_samples_seen = epoch * num_model * num_samples_per_model * num_images_per_model + samples_count
    time_per_sample = time_passed/num_samples_seen
    estimate_total_time = time_per_sample * num_epoch * num_model * num_samples_per_model * num_images_per_model
    estimate_time_left = estimate_total_time - time_passed

    return estimate_time_left


if __name__ == '__main__':
    print("Loading parameters...")

    # load parameters
    param_all = json.load(open(PARAM_FILE))
    param_dec = param_all["decoder"]
    param_enc = param_all["encoder"]
    param_vae = param_all["vae"]
    resolution = param_all["resolution_used_for_training"]
    latent_size = param_all["latent_size"]

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

    ######################################## only used for testing ########################################
    list_model_hash = list_model_hash[:5]
    ######################################## only used for testing ########################################

    num_model = len(list_model_hash)
    num_images_per_model = len(annotations[list_model_hash[0]])

    ######################################## only used for testing ########################################
    num_images_per_model = 10
    ######################################## only used for testing ########################################

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


    # Init training dataset
    training_set = DatasetVAE(list_model_hash, dict_gt_data, annotations, num_images_per_model, num_samples_per_model, param_enc["image"], param_enc["network"], IMAGES_PATH, MATRIX_PATH)
    training_generator= torch.utils.data.DataLoader(training_set, **param_vae["dataLoader"])

    
    # fill a xyz grid to give as input to the decoder 
    xyz = init_xyz(resolution)

    # Init decoder and encoder
    encoder = EncoderGrid(latent_size, param_enc["network"]).cuda()
    decoder = Decoder(latent_size, batch_norm=False)

    encoder.apply(init_weights)
    decoder.apply(init_weights)

    # initialize optimizer and scheduler
    optimizer, scheduler = init_opt_sched(encoder, decoder, param_vae["optimizer"])


    encoder.train()
    decoder.train()

    print(f"Start trainging... with {num_model} models")

    time_start = time.time()
    
    for epoch in range(param_vae["num_epoch"]):
        samples_count = 0
        for batch_grid, batch_sdf_gt, batch_rgb_gt, batch_xyz_idx in training_generator:
            optimizer.zero_grad()

            batch_size = len(batch_grid)

            # transfer to gpu
            batch_grid = batch_grid.cuda()
            sdf_gt = sdf_gt.cuda()
            rgb_gt = rgb_gt.cuda()
            batch_xyz_idx = torch.tensor(batch_xyz_idx)


            predicted_code = encoder(batch_grid)

            pred = decoder(predicted_code, xyz[batch_xyz_idx])
            pred_sdf = pred[:,0]
            pred_rgb = pred[:,1:]

            loss_sdf, loss_rgb, loss_kl = compute_loss(pred_sdf, pred_rgb, sdf_gt, rgb_gt, predicted_code, threshold_precision, param_dec)
            # loss_sdf, loss_rgb, loss_kl = compute_loss(pred_sdf, pred_rgb, sdf_gt, rgb_gt, predicted_code, predicted_code_log_std, threshold_precision, param_dec)

            loss_total = loss_sdf + loss_rgb + loss_kl


            #update weights
            loss_total.backward()
            optimizer.step()

            # estime time left
            samples_count += batch_size
            time_left = compute_time_left(time_start, samples_count, num_model, num_samples_per_model, num_images_per_model, epoch, param_vae["num_epoch"])

             # print everyl X model seen
            if samples_count%(param_vae["num_batch_between_print"] * batch_size) == 0:

                #log
                # logs["total"].append(loss_total.detach().cpu())
                # logs["sdf"].append(loss_sdf.detach().cpu())
                # logs["rgb"].append(loss_rgb.detach().cpu())
                # logs["reg"].append(loss_kl.detach().cpu())

                # print("Epoch {} / {:.2f}% ,loss: sdf: {:.5f}, rgb: {:.5f}, reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, min/max rgb: {:.2f}/{:.2f}, code std/mu: {:.2f}/{:.2f}, time left: {} min".format(\
                #     epoch, 100 * samples_count / (num_model * num_samples_per_model), loss_sdf, loss_rgb, loss_kl, \
                #     pred_sdf.min() * resolution, pred_sdf.max() * resolution, pred_rgb.min() * 255, pred_rgb.max() * 255, \
                #     (lat_code_log_std.weight.exp()).mean(), (lat_code_mu.weight).abs().mean(), (int)(time_left/60)))

                print("Epoch {} / {:.2f}% ,loss: sdf: {:.5f}, rgb: {:.5f}, reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, min/max rgb: {:.2f}/{:.2f}, time left: {} min".format(\
                    epoch, 100 * samples_count / (num_model * num_samples_per_model), loss_sdf, loss_rgb, loss_kl, \
                    pred_sdf.min() * resolution, pred_sdf.max() * resolution, pred_rgb.min() * 255, pred_rgb.max() * 255, (int)(time_left/60)))

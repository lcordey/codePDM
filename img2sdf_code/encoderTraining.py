from math import log
import numpy as np
import torch
import torch.nn as nn
import yaml
import pickle
import time
from skimage import color

from marching_cubes_rgb import *
from utils import *

from networks import EncoderGrid
from dataLoader import DatasetGrid

import IPython


ENCODER_PATH = "models_and_codes/encoderGrid.pth"
DECODER_PATH = "models_and_codes/decoder.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
PARAM_FILE = "config/param.yaml"
PARAM_SAVE_FILE = "config/param_encoder.yaml"
VEHICLE_VALIDATION_PATH = "config/vehicle_validation.txt"
ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
LOGS_PATH = "../../image2sdf/logs/encoder/log.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"
MATRIX_PATH = "../../image2sdf/input_images/matrix_w2c.pkl"


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        torch.nn.init.xavier_uniform_(m.weight)
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
    param_all = yaml.safe_load(open(PARAM_FILE))
    resolution = param_all["resolution_used_for_training"]
    param = param_all["encoder"]

    # Load decoder
    decoder = torch.load(DECODER_PATH).cuda()
    decoder.eval()

    # load codes and annotations
    dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))
    annotations = pickle.load(open(ANNOTATIONS_PATH, 'rb'))
    
    with open(VEHICLE_VALIDATION_PATH) as f:
        list_hash_validation = f.read().splitlines()
    list_hash_validation = list(list_hash_validation)

    # Only consider model which appear in both annotation and code and that are not used for validation
    list_hash = []
    for hash in annotations.keys():
        if hash in dict_hash_2_code.keys():
            if hash not in list_hash_validation:
                list_hash.append(hash)

    num_model = len(list_hash)
    num_images_per_model = len(annotations[list_hash[0]])

    # Init training dataset
    training_set = DatasetGrid(list_hash, annotations, num_images_per_model, param["image"], param["network"], IMAGES_PATH, MATRIX_PATH)
    training_generator= torch.utils.data.DataLoader(training_set, **param["dataLoader"])

    validation_set = DatasetGrid(list_hash_validation, annotations, num_images_per_model, param["image"], param["network"], IMAGES_PATH, MATRIX_PATH)
    validation_generator= torch.utils.data.DataLoader(validation_set, **param["dataLoaderValidation"])

    # Init Encoder
    encoder = EncoderGrid(param_all["latent_size"], param["network"]).cuda()
    encoder.apply(init_weights)

    # initialize optimizer and scheduler
    optimizer, scheduler = init_opt_sched(encoder, param["optimizer"])
    loss = torch.nn.MSELoss()

    # fill a xyz grid to give as input to the decoder for validation
    xyz = init_xyz(resolution)


    # logs
    logs = dict()

    logs["training"] = []

    logs["validation"] = dict()
    logs["validation"]["l2"] = []
    logs["validation"]["sdf"] = []
    logs["validation"]["rgb"] = []
    logs["validation"]["lab"] = []
    logs["validation"]["cham_sdf"] = []
    logs["validation"]["cham_rgb"] = []
    logs["validation"]["cham_lab"] = []


    encoder.train()
    print(f"Start trainging... with {num_model} models")

    time_start = time.time()
    
    for epoch in range(param["num_epoch"]):
        samples_count = 0
        for batch_grid, batch_model_hash in training_generator:

            optimizer.zero_grad()
            batch_size = len(batch_grid)

            # transfer to gpu
            batch_grid = batch_grid.cuda()

            # get target code
            target_code = torch.empty([batch_size, param_all["latent_size"]]).cuda()
            for model_hash, i in zip(batch_model_hash, range(batch_size)):
                target_code[i] = dict_hash_2_code[model_hash]


            predicted_code = encoder(batch_grid)

            # compute loss
            loss_training = loss(predicted_code, target_code)

            #update weights
            loss_training.backward()
            optimizer.step()

            # compute time left
            samples_count += batch_size
            time_left = compute_time_left(time_start, samples_count, num_model, num_images_per_model, epoch, param["num_epoch"])

            # print everyl X model seen
            if samples_count%(param["num_batch_between_print"] * batch_size) == 0:

                logs["training"].append(loss_training.detach().cpu())
                print("epoch: {}/{:.2f}%, L2 loss: {:.5f}, L1 loss: {:.5f} mean abs pred: {:.5f}, mean abs target: {:.5f}, LR: {:.6f}, time left: {} min".format(\
                    epoch, 100 * samples_count / (num_model * num_images_per_model), loss_training, \
                    abs(predicted_code - target_code).mean(), abs(predicted_code).mean(), abs(target_code).mean(),\
                    optimizer.param_groups[0]['lr'],  (int)(time_left/60) ))


            # validation 
            if samples_count%(param["num_batch_between_validation"] * batch_size) == 0 or samples_count == batch_size:

                encoder.eval()

                log_l2_val = []
                log_sdf = []
                log_rgb = []
                log_lab = []
                log_cham_sdf = []
                log_cham_rgb = []
                log_cham_lab = []

                samples_count_val = 0
                for images_val, model_hash_val in validation_generator:

                    # transfer to gpu
                    images_val = images_val.cuda()
                    target_code_val = dict_hash_2_code[model_hash_val[0]].unsqueeze(0).cuda() # -> [0] because batch size should always be 1 for validation

                    # compute predicted code
                    predicted_code_val = encoder(images_val)

                    # compute loss
                    loss_l2= (predicted_code_val-target_code_val).norm()
                    log_l2_val.append(loss_l2.detach().cpu())

                    # compute the sdf from codes 
                    sdf_validation = decoder(predicted_code_val.repeat_interleave(resolution * resolution * resolution, dim=0),xyz).detach()
                    sdf_target= decoder(target_code_val.repeat_interleave(resolution * resolution * resolution, dim=0),xyz).detach()

                    # assign weight of 0 for easy samples that are well trained
                    threshold_precision = 1/resolution
                    weight_sdf = ~((sdf_validation[:,0] > threshold_precision).squeeze() * (sdf_target[:,0] > threshold_precision).squeeze()) \
                        * ~((sdf_validation[:,0] < -threshold_precision).squeeze() * (sdf_target[:,0] < -threshold_precision).squeeze())

                    # loss l1 in distance error per samples
                    loss_sdf = torch.nn.L1Loss(reduction='none')(sdf_validation[:,0].squeeze(), sdf_target[:,0])
                    loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()
                    loss_sdf *= resolution
                
                    # loss rgb in pixel value difference per color per samples
                    rgb_gt_normalized = sdf_target[:,1:]
                    loss_rgb = torch.nn.L1Loss(reduction='none')(sdf_validation[:,1:], rgb_gt_normalized)
                    loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean()/3 * weight_sdf.numel()/weight_sdf.count_nonzero()
                    loss_rgb *= 255

                    # lab loss
                    lab_validation = sdf_validation[:,1:].copy() / 255
                    lab_validation = torch.tensor(color.rgb2lab(lab_validation.cpu())).cuda()

                    lab_target = sdf_target[:,1:].copy() / 255
                    lab_target = torch.tensor(color.rgb2lab(lab_target.cpu())).cuda()

                    # loss LAB in pixel value difference per color per samples
                    loss_lab = torch.nn.L1Loss(reduction='none')(lab_validation, lab_target)
                    loss_lab = ((loss_lab[:,0] * weight_sdf) + (loss_lab[:,1] * weight_sdf) + (loss_lab[:,2] * weight_sdf)).mean()/3 * weight_sdf.numel()/weight_sdf.count_nonzero()

                    # save losses
                    log_sdf.append(loss_sdf.detach().cpu())
                    log_rgb.append(loss_rgb.detach().cpu())
                    log_lab.append(loss_lab.detach().cpu())


                    # compute chamfer losses
                    sdf_target = sdf_target.reshape(resolution, resolution, resolution, 4)
                    if(np.min(sdf_target[:,:,:,0]) < 0 and np.max(sdf_target[:,:,:,0]) > 0):
                        vertices_target, faces_target = marching_cubes(sdf_target[:,:,:,0])
                        colors_v_target = exctract_colors_v(vertices_target, sdf_target)

                    vertices_target = torch.tensor(vertices_target.copy())
                    colors_v_target = torch.tensor(colors_v_target/255).unsqueeze(0).cuda()

                    sdf_validation = sdf_validation.reshape(resolution, resolution, resolution, 4)
                    if(np.min(sdf_validation[:,:,:,0]) < 0 and np.max(sdf_validation[:,:,:,0]) > 0):
                        vertices_validation, faces_validation = marching_cubes(sdf_validation[:,:,:,0])
                        colors_v_validation = exctract_colors_v(vertices_validation, sdf_validation)

                    vertices_validation = torch.tensor(vertices_validation.copy())
                    colors_v_validation = torch.tensor(colors_v_validation/255).unsqueeze(0).cuda()

                    cham_sdf, cham_rgb, cham_lab = chamfer_distance_rgb(vertices_validation, vertices_target, colors_x = colors_v_validation, colors_y = colors_v_target)

                    log_cham_sdf.append(cham_sdf)
                    log_cham_rgb.append(cham_rgb)
                    log_cham_lab.append(cham_lab)

                    samples_count_val += 1
                    if samples_count_val == param["num_images_validation"] :
                        break

                loss_l2_val = torch.tensor(log_l2_val).mean()
                loss_sdf_val = torch.tensor(log_sdf).mean()
                loss_rgb_val = torch.tensor(log_rgb).mean()
                loss_lab_val = torch.tensor(log_lab).mean()
                error_cham_sdf = torch.tensor(log_cham_sdf).mean()
                error_cham_rgb = torch.tensor(log_cham_rgb).mean()
                error_cham_lab = torch.tensor(log_cham_lab).mean()

                logs["validation"]["l2"].append(loss_l2_val)
                logs["validation"]["sdf"].append(loss_sdf_val)
                logs["validation"]["rgb"].append(loss_rgb_val)
                logs["validation"]["lab"].append(loss_lab_val)
                logs["validation"]["cham_sdf"].append(error_cham_sdf)
                logs["validation"]["cham_rgb"].append(error_cham_rgb)
                logs["validation"]["cham_lab"].append(error_cham_lab)


                print("\n****************************** VALIDATION ******************************")

                print(f"l2 predicted code error: {loss_l2_val:2.3f}")
                print(f"sdf error: {loss_sdf_val:2.3f}")
                print(f"rgb error: {loss_rgb_val:2.3f}")
                print(f"lab error: {loss_rgb_val:2.3f}")
                print(f"cham sdf error: {error_cham_sdf:2.3f}")
                print(f"cham rgb error: {error_cham_rgb:2.3f}")
                print(f"cham lab error: {error_cham_lab:2.3f}")

                print("****************************** VALIDATION ******************************\n")

                
                encoder.train()


        scheduler.step()

        if epoch%5==0:
            # save encoder
            torch.save(encoder, ENCODER_PATH)

            # save logs
            with open(LOGS_PATH, "wb") as fp:
                pickle.dump(logs, fp)
                
            # save param used
            with open(PARAM_SAVE_FILE, 'w') as file:
                yaml.dump(param_all, file)


    print(f"Training finish in {(int)((time.time() - time_start) / 60)} min")


    ###### Saving eNcoder ######
    # save encoder
    torch.save(encoder, ENCODER_PATH)

    # save logs
    with open(LOGS_PATH, "wb") as fp:
        pickle.dump(logs, fp)
        
    
    # save param used
    with open(PARAM_SAVE_FILE, 'w') as file:
        yaml.dump(param_all, file)



import numpy as np
import torch
import pickle
import glob
import yaml
import time
import h5py
import os
import matplotlib.pyplot as plt

from networks import Decoder
from dataLoader import DatasetDecoderSDF

import IPython

# INPUT FILE
SDF_DIR = "../../img_supervision/sdf/"
IMAGES_PATH = "../../img_supervision/input_images/images/"
ANNOTATIONS_PATH = "../../img_supervision/input_images/annotations.pkl"
PARAM_FILE = "config/param.yaml"

# SAVE FILE
DECODER_SDF_PATH = "models_and_codes/decoder_sdf.pth"
PARAM_SAVE_FILE = "config/param_decoder.yaml"
LOGS_PATH = "../../img_supervision/logs/decoder/log.pkl"
DICT_HASH_2_IDX_PATH = "models_and_codes/dict_hash_2_idx.pkl"
DICT_HASH_2_CODE_PATH = "models_and_codes/dict_hash_2_code.pkl"
LATENT_CODE_MU_PATH = "models_and_codes/latent_code_mu.pkl"
LATENT_CODE_LOG_STD_PATH = "models_and_codes/latent_code_log_std.pkl"
LIST_MODELS_PATH = "models_and_codes/list_models.pkl"



def init_xyz(resolution):
    """ fill 3d grid representing 3d location to give as input to the decoder """
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz

def init_lat_codes(num_scenes, latent_size):
    """initialize random latent code for every model"""

    lat_code_mu = torch.nn.Embedding(num_scenes, latent_size).cuda()
    torch.nn.init.normal_(
        lat_code_mu.weight.data,
        0.0,
        1.0,
    )
    lat_code_log_std = torch.nn.Embedding(num_scenes, latent_size).cuda()
    torch.nn.init.normal_(
        lat_code_log_std.weight.data,
        0.0,
        0.0,
    )

    return lat_code_mu, lat_code_log_std

def init_opt_sched(decoder, lat_code_mu, lat_code_log_std, param):
    """ initialize optimizer and scheduler"""

    optimizer_decoder = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": param["eta_decoder"],
            },
        ]
    )
    optimizer_code = torch.optim.Adam(
        [
            {
                "params": lat_code_mu.parameters(),
                "lr": param["eta_latent_space_mu"],
            },
            {
                "params": lat_code_log_std.parameters(),
                "lr": param["eta_latent_space_std"],
            },
        ]
    )

    scheduler_decoder = torch.optim.lr_scheduler.ExponentialLR(optimizer_decoder, gamma=param["gamma_decoder_LR"])
    scheduler_code = torch.optim.lr_scheduler.ExponentialLR(optimizer_code, gamma=param["gamma_code_LR"])

    return optimizer_decoder, optimizer_code, scheduler_decoder, scheduler_code

def compute_time_left(time_start, samples_count, num_model, num_samples_per_model, epoch, num_epoch):
    """ Compute time left until the end of training """
    time_passed = time.time() - time_start
    num_samples_seen = epoch * num_model * num_samples_per_model + samples_count
    time_per_sample = time_passed/num_samples_seen
    estimate_total_time = time_per_sample * num_epoch * num_model * num_samples_per_model
    estimate_time_left = estimate_total_time - time_passed

    return estimate_time_left

def compute_loss_sdf(pred_sdf, sdf_gt, threshold_precision, lambda_sdf):
    """ compute sdf, rgb and regression loss """

    loss = torch.nn.MSELoss(reduction='none')

    # assign weight of 0 for easy samples that are well trained
    weight_sdf = ~((pred_sdf > threshold_precision).squeeze() * (sdf_gt > threshold_precision).squeeze()) \
        * ~((pred_sdf < -threshold_precision).squeeze() * (sdf_gt < -threshold_precision).squeeze())

    #L2 loss, only for hard samples
    loss_sdf = loss(pred_sdf.squeeze(), sdf_gt)
    loss_sdf = (loss_sdf * weight_sdf).sum()/weight_sdf.count_nonzero()
    loss_sdf *= lambda_sdf

    return loss_sdf

def compute_loss_reg(lat_code_mu, lat_code_log_std, lambda_kl):

    # regularization loss
    loss_kl = (-0.5 * (1 + lat_code_log_std.weight - lat_code_mu.weight.pow(2) - lat_code_log_std.weight.exp())).mean()
    loss_kl *= lambda_kl

    return loss_kl


if __name__ == '__main__':
    print("Loading parameters...")

    # load parameters
    param_all = yaml.safe_load(open(PARAM_FILE))
    param_sdf = param_all["decoder_sdf"]
    resolution = param_all["resolution_used_for_training"]
 
    threshold_precision = 1.0/resolution
    num_samples_per_model = resolution * resolution * resolution

    # fill a xyz grid to give as input to the decoder 
    xyz = init_xyz(resolution)
    
    # get models' hashs
    list_model_hash = []
    for val in glob.glob(SDF_DIR + "*.h5"):
        list_model_hash.append(os.path.basename(val).split('.')[0])

    ######################################## only used for testing ########################################
    # list_model_hash = list_model_hash[:50]
    list_model_hash = list_model_hash[:10]
    ######################################## only used for testing ########################################


    num_model = len(list_model_hash)
    
    # initialize a random latent code for each models
    lat_code_mu, lat_code_log_std = init_lat_codes(num_model, param_all["latent_size"])

    # create a dictionary going from an hash to a corresponding index
    idx = torch.arange(num_model).type(torch.LongTensor)
    dict_model_hash_2_idx = dict()

    # dict for training model
    for model_hash, i in zip(list_model_hash, range(num_model)):
        dict_model_hash_2_idx[model_hash] = idx[i]


    # load every models
    print("Loading models...")
    dict_gt_data = dict()
    dict_gt_data["sdf"] = dict()

    # load training data in dict
    for model_hash, i in zip(list_model_hash, range(num_model)):
        if i%25 == 0:
            print(f"loading models: {i}/{num_model:3.0f}")

        # load sdf tensor
        h5f = h5py.File(SDF_DIR + model_hash + '.h5', 'r')
        h5f_tensor = torch.tensor(h5f["tensor"][()], dtype = torch.float)

        # split sdf and rgb then reshape
        sdf_gt = np.reshape(h5f_tensor[:,:,:,0], [num_samples_per_model])

        # normalize
        sdf_gt = sdf_gt / resolution
        
        # store in dict
        dict_gt_data["sdf"][model_hash] = sdf_gt


    # Init dataset and dataloader
    training_dataset = DatasetDecoderSDF(list_model_hash, dict_gt_data, num_samples_per_model, dict_model_hash_2_idx)
    training_generator = torch.utils.data.DataLoader(training_dataset, **param_sdf["dataLoader"])

    # initialize decoder
    decoder_sdf = Decoder(param_all["latent_size"], "sdf", batch_norm=True).cuda()

    # initialize optimizer and scheduler
    optimizer_decoder, optimizer_code, scheduler_decoder, scheduler_code = init_opt_sched(decoder_sdf, lat_code_mu, lat_code_log_std, param_sdf["optimizer"])

    # logs
    logs = dict()
    logs["total"] = []
    logs["sdf"] = []
    logs["reg"] = []
    logs["l2_dup"] = []
    logs["l2_rand"] = []

    print("Start training sdf ...")
    decoder_sdf.train()

    time_start = time.time()

    for epoch in range (param_sdf["num_epoch"]):
        samples_count = 0
        for model_idx, sdf_gt, xyz_idx in training_generator:
            optimizer_decoder.zero_grad()
            optimizer_code.zero_grad()

            batch_size = len(model_idx)

            # transfer to gpu
            sdf_gt = sdf_gt.cuda()
            model_idx = model_idx.cuda()
            xyz_idx = xyz_idx

            # Compute latent code 
            coeff_std = torch.empty(batch_size, param_all["latent_size"]).normal_().cuda()
            latent_code = coeff_std * lat_code_log_std(model_idx).exp() * param_sdf["lambda_variance"] + lat_code_mu(model_idx)

            # get sdf from decoder
            pred = decoder_sdf(latent_code, xyz[xyz_idx])
            pred_sdf = pred[:,0]

            # compute loss
            loss_sdf = compute_loss_sdf(pred_sdf, sdf_gt, threshold_precision, param_sdf["lambda_sdf"])
            loss_kl = compute_loss_reg(lat_code_mu, lat_code_log_std, param_sdf["lambda_kl"])
            loss_total = loss_sdf + loss_kl

            #update weights
            loss_total.backward()
            # optimizer.step()
            optimizer_decoder.step()
            optimizer_code.step()

            samples_count += batch_size

            # print everyl X model seen
            if samples_count%(param_sdf["num_batch_between_print"] * batch_size) == 0:

                # estime time left
                time_left = compute_time_left(time_start, samples_count, num_model, num_samples_per_model, epoch, param_sdf["num_epoch"])

                print("Epoch {} / {:.2f}% ,loss: sdf: {:.5f}, reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, code std/mu: {:.2f}/{:.2f}, time left: {} min".format(\
                    epoch, 100 * samples_count / (num_model * num_samples_per_model), loss_sdf, loss_kl, \
                    pred_sdf.min() * resolution, pred_sdf.max() * resolution, \
                    (lat_code_log_std.weight.exp()).mean(), (lat_code_mu.weight).abs().mean(), (int)(time_left/60)))


                #log
                logs["total"].append(loss_total.detach().cpu())
                logs["sdf"].append(loss_sdf.detach().cpu())
                logs["reg"].append(loss_kl.detach().cpu())
                
                
        scheduler_decoder.step()
        scheduler_code.step()

    print(f"Training sdf done in {(int)((time.time() - time_start) / 60)} min")

    # SAVE EVERYTHING
    torch.save(decoder_sdf, DECODER_SDF_PATH)

    # save latent code in dict
    dict_hash_2_code = dict()
    for model_hash in list_model_hash:
        dict_hash_2_code[model_hash] = lat_code_mu(dict_model_hash_2_idx[model_hash].cuda()).detach().cpu()

    with open(DICT_HASH_2_CODE_PATH, "wb") as file:
        pickle.dump(dict_hash_2_code, file)

    with open(LIST_MODELS_PATH, "wb") as file:
        pickle.dump(list_model_hash, file)

    with open(LATENT_CODE_MU_PATH, "wb") as file:
        pickle.dump(lat_code_mu, file)

    with open(LATENT_CODE_LOG_STD_PATH, "wb") as file:
        pickle.dump(lat_code_log_std, file)

    with open(DICT_HASH_2_IDX_PATH, "wb") as file:
        pickle.dump(dict_model_hash_2_idx, file)

    # save logs
    with open(LOGS_PATH, "wb") as file:
        pickle.dump(logs, file)

    # save param used
    with open(PARAM_SAVE_FILE, 'w') as file:
        yaml.dump(param_all, file)

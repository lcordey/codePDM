import numpy as np
import torch
import pickle
import glob
import json
import time

from networks import Decoder
from dataLoader import DatasetDecoder
from marching_cubes_rgb import *

import IPython

# directory which contain the SDF input and where all the output will be generated
MAIN_DIR = "../../image2sdf/"
# MAIN_DIR = "/home/loic/MasterPDM/image2sdf/"

DECODER_PATH = "models_and_codes/decoderSDF.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
LOGS_PATH = "../../image2sdf/logs/log.pkl"
PARAM_FILE = "config/param.json"

SDF_DIR = MAIN_DIR + "sdf/"
RESOLUTION = 64



def init_xyz(resolution):
    """ fill 3d grid representing 3d location to give as input to the decoder """
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz

def init_lat_vecs(num_scenes, latent_size):
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

def init_opt_sched(decoder, lat_vecs_mu, lat_vecs_log_std, param):
    """ initialize optimizer and scheduler"""

    optimizer = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": param["eta_decoder"],
            },
            {
                "params": lat_vecs_mu.parameters(),
                "lr": param["eta_latent_space_mu"],
            },
            {
                "params": lat_vecs_log_std.parameters(),
                "lr": param["eta_latent_space_std"],
            },
        ]
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=param["gammaLR"])

    return optimizer, scheduler

def compute_time_left(time_start, model_count, num_model, epoch, num_epoch):
    """ Compute time left until the end of training """
    time_passed = time.time() - time_start
    num_model_seen = epoch * num_model + model_count
    time_per_model = time_passed/num_model_seen
    estimate_total_time = time_per_model * num_epoch * num_model
    estimate_time_left = estimate_total_time - time_passed

    return estimate_time_left

def compute_loss(pred_sdf, pred_rgb, sdf_gt, rgb_gt, threshold_precision, param):
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
    loss_kl = (-0.5 * (1 + lat_code_log_std.weight - lat_code_mu.weight.pow(2) - lat_code_log_std.weight.exp())).mean()
    loss_kl *= param["lambda_kl"]

    return loss_sdf, loss_rgb, loss_kl

if __name__ == '__main__':
    print("Loading parameters...")

    # load parameters
    param_all = json.load(open(PARAM_FILE))
    param = param_all["decoder"]

    resolution = RESOLUTION
    threshold_precision = 1.0/resolution
    # num_samples_per_model = resolution * resolution * resolution
    num_samples_per_model = param["num_samples_per_model"]
    batch_size = param["dataLoader"]["batch_size"]

    # get models' hashs
    list_model_hash = []
    for val in glob.glob(SDF_DIR + "*.h5"):
        list_model_hash.append(os.path.basename(val).split('.')[0])
    num_model = len(list_model_hash)

    # dataLoader for training dataset
    training_dataset = DatasetDecoder(list_model_hash, SDF_DIR, resolution, num_samples_per_model)
    training_generator = torch.utils.data.DataLoader(training_dataset, **param["dataLoader"])

    # fill a xyz grid to give as input to the decoder 
    xyz = init_xyz(resolution)

    # initialize a random latent code for each models
    lat_code_mu, lat_code_log_std = init_lat_vecs(num_model, param["latent_size"])

    # create a dictionary going from an hash to a corresponding index
    idx = torch.arange(num_model).type(torch.LongTensor).cuda()
    dict_model_hash_2_idx = dict()
    for model_hash, i in zip(list_model_hash, range(num_model)):
        dict_model_hash_2_idx[model_hash] = idx[i]

    # initialize decoder
    decoder = Decoder(param["latent_size"]).cuda()

    # initialize optimizer and scheduler
    optimizer, scheduler = init_opt_sched(decoder, lat_code_mu, lat_code_log_std, param["optimizer"])


    # logs
    logs = dict()
    logs["total"] = []
    logs["sdf"] = []
    logs["rgb"] = []
    logs["reg"] = []

    print("Start training...")
    decoder.train()

    time_start = time.time()

    for epoch in range (param["num_epoch"]):
        model_count = 0
        for hash, sdf_gt, rgb_gt in training_generator:
            optimizer.zero_grad()

            print(time.time() - time_start)

            # only 1 sample per batch!
            # hash = hash[0]
            # sdf_gt = sdf_gt[0]
            # rgb_gt = rgb_gt[0]

            # transfer to gpu
            sdf_gt = sdf_gt.cuda()
            rgb_gt = rgb_gt.cuda()

            ##### compute sdf prediction #####
            # code_mu, code_log_std = lat_code_mu(dict_model_hash_2_idx[hash]), lat_code_log_std(dict_model_hash_2_idx[hash])
            # latent_code =  torch.empty(num_samples_per_model, param["latent_size"]).normal_().cuda() * code_log_std.exp() * param["lambda_variance"] + code_mu

            # pred = decoder(latent_code, xyz)

            code_log_std = []
            code_mu = []
            latent_code_list = []
            a = torch.empty(batch_size, param["latent_size"]).normal_().cuda()

            for i in range(len(hash)):
                code_mu.append(lat_code_mu(dict_model_hash_2_idx[hash[i]]))
                code_log_std.append(lat_code_log_std(dict_model_hash_2_idx[hash[i]]))
                latent_code_list.append(a[i] * code_log_std[i].exp() * param["lambda_variance"] + code_mu[i])

            latent_code = torch.empty([len(latent_code_list), param["latent_size"]]).cuda()
            for i in range(len(latent_code)):
                latent_code[i] = latent_code_list[i]

            pred = decoder(latent_code.repeat_interleave(num_samples_per_model, dim=0), xyz[:num_samples_per_model].repeat(batch_size,1))

            ##### compute loss and store logs #####
            pred_sdf = pred[:,0]
            pred_rgb = pred[:,1:]
            loss_sdf, loss_rgb, loss_kl = compute_loss(pred_sdf, pred_rgb, sdf_gt.reshape(batch_size * num_samples_per_model), rgb_gt.reshape(batch_size * num_samples_per_model, 3), threshold_precision, param)
            
            loss_total = loss_sdf + loss_rgb + loss_kl

            #log
            logs["total"].append(loss_total.detach().cpu())
            logs["sdf"].append(loss_sdf.detach().cpu())
            logs["rgb"].append(loss_rgb.detach().cpu())
            logs["reg"].append(loss_kl.detach().cpu())

            # #update weights
            # loss_total.backward()
            # optimizer.step()

            # estime time left
            model_count += 1
            time_left = compute_time_left(time_start, model_count, num_model, epoch, param["num_epoch"])

            # print
            print("Epoch {} / {:.2f}% ,loss: sdf: {:.5f}, rgb: {:.5f}, reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, min/max rgb: {:.2f}/{:.2f}, code std/mu: {:.2f}/{:.2f}, time left: {} min".format(\
                epoch, model_count / num_model * 100, loss_sdf, loss_rgb, loss_kl, \
                pred_sdf.min() * resolution, pred_sdf.max() * resolution, pred_rgb.min() * 255, pred_rgb.max() * 255, \
                (lat_code_log_std.weight.exp()).mean(), (lat_code_mu.weight).abs().mean(), (int)(time_left/60)))
                
            if (loss_sdf.isnan()):
                IPython.embed()


            #update weights
            loss_total.backward()
            optimizer.step()

            time_start = time.time()

        scheduler.step()

    print(f"Training finish in {(int)((time.time() - time_start) / 60)} min")


    with open(LOGS_PATH, "wb") as fp:
        pickle.dump(logs, fp)
    
    
    dict_hash_2_code = dict()
    for model_hash in list_model_hash:
        dict_hash_2_code[model_hash] = lat_code_mu(dict_model_hash_2_idx[hash])

    with open(LATENT_CODE_PATH, "wb") as fp:
        pickle.dump(logs, fp)



 
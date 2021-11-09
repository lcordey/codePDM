from PIL.Image import radial_gradient
import numpy as np
import torch
import pickle
import glob
# import json
import yaml
import time

from networks import Decoder
from dataLoader import DatasetDecoder
from marching_cubes_rgb import *

import IPython


DECODER_PATH = "models_and_codes/decoder.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
# PARAM_FILE = "config/param.json"
PARAM_FILE = "config/param.yaml"
PARAM_SAVE_FILE = "config/param_decoder.yaml"
LOGS_PATH = "../../image2sdf/logs/decoder/log.pkl"
SDF_DIR = "../../image2sdf/sdf/"
# SDF_DIR = "../../image2sdf/sdf_duplicate/"

num_model_duplicate = 20



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

def init_opt_sched(decoder, lat_vecs_mu, lat_vecs_log_std, param):
    """ initialize optimizer and scheduler"""

    # optimizer = torch.optim.Adam(
    #     [
    #         {
    #             "params": decoder.parameters(),
    #             "lr": param["eta_decoder"],
    #         },
    #         {
    #             "params": lat_vecs_mu.parameters(),
    #             "lr": param["eta_latent_space_mu"],
    #         },
    #         {
    #             "params": lat_vecs_log_std.parameters(),
    #             "lr": param["eta_latent_space_std"],
    #         },
    #     ]
    # )

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
                "params": lat_vecs_mu.parameters(),
                "lr": param["eta_latent_space_mu"],
            },
            {
                "params": lat_vecs_log_std.parameters(),
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

def compute_loss(pred_sdf, pred_rgb, sdf_gt, rgb_gt, lat_code_mu, lat_code_log_std, threshold_precision, param):
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
    # param_all = json.load(open(PARAM_FILE))
    param_all = yaml.safe_load(open(PARAM_FILE))
    param = param_all["decoder"]
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
    # list_model_hash = list_model_hash[:80]
    ######################################## only used for testing ########################################

    # create duplicated models
    list_model_hash_dup = []
    for model_hash, i in zip(list_model_hash, range(num_model_duplicate)):
        list_model_hash_dup.append(model_hash + "_dup")

    num_model_training= len(list_model_hash)
    num_model_total = num_model_training + num_model_duplicate
    
    # initialize a random latent code for each models
    lat_code_mu, lat_code_log_std = init_lat_codes(num_model_total, param_all["latent_size"])

    # create a dictionary going from an hash to a corresponding index
    idx = torch.arange(num_model_total).type(torch.LongTensor)
    dict_model_hash_2_idx = dict()

    # idx for training model
    for model_hash, i in zip(list_model_hash, range(num_model_training)):
        dict_model_hash_2_idx[model_hash] = idx[i]

    # idx for duplicated models
    for model_hash_dup, i in zip(list_model_hash_dup, range(num_model_duplicate)):
        dict_model_hash_2_idx[model_hash_dup] = idx[i + num_model_training]

    # load every models
    print("Loading models...")
    dict_gt_data = dict()
    dict_gt_data["sdf"] = dict()
    dict_gt_data["rgb"] = dict()

    # load training data in dict
    for model_hash, i in zip(list_model_hash, range(num_model_training)):
        if i%25 == 0:
            print(f"loading models: {i}/{num_model_training:3.0f}")

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

    # load duplicate data in dict
    for model_hash, model_hash_dup, i in zip(list_model_hash, list_model_hash_dup, range(num_model_duplicate)):
        if i%25 == 0:
            print(f"loading models: {i}/{num_model_duplicate:3.0f}")

        # store in dict
        dict_gt_data["sdf"][model_hash_dup] = dict_gt_data["sdf"][model_hash]
        dict_gt_data["rgb"][model_hash_dup] = dict_gt_data["rgb"][model_hash]

    # Init dataset and dataloader
    training_dataset = DatasetDecoder(list_model_hash + list_model_hash_dup, dict_gt_data, num_samples_per_model, dict_model_hash_2_idx)
    training_generator = torch.utils.data.DataLoader(training_dataset, **param["dataLoader"])

    # initialize decoder
    decoder = Decoder(param_all["latent_size"], batch_norm=True).cuda()

    # initialize optimizer and scheduler
    # optimizer, scheduler = init_opt_sched(decoder, lat_code_mu, lat_code_log_std, param["optimizer"])
    optimizer_decoder, optimizer_code, scheduler_decoder, scheduler_code = init_opt_sched(decoder, lat_code_mu, lat_code_log_std, param["optimizer"])

    # logs
    logs = dict()
    logs["total"] = []
    logs["sdf"] = []
    logs["rgb"] = []
    logs["reg"] = []
    logs["l2_dup"] = []
    logs["l2_rand"] = []

    print("Start training...")
    decoder.train()

    time_start = time.time()

    for epoch in range (param["num_epoch"]):
        samples_count = 0
        for model_idx, sdf_gt, rgb_gt, xyz_idx in training_generator:
        #     optimizer.zero_grad()
            optimizer_decoder.zero_grad()
            optimizer_code.zero_grad()

            # time_loading = time.time() - time_start
            # print(f"Time to load the data: {time_loading}")
            # time_start = time.time()

            batch_size = len(model_idx)

            # transfer to gpu
            sdf_gt = sdf_gt.cuda()
            rgb_gt = rgb_gt.cuda()
            model_idx = model_idx.cuda()
            xyz_idx = xyz_idx

            # Compute latent code 
            coeff_std = torch.empty(batch_size, param_all["latent_size"]).normal_().cuda()
            latent_code = coeff_std * lat_code_log_std(model_idx).exp() * param["lambda_variance"] + lat_code_mu(model_idx)

            # get sdf from decoder
            pred = decoder(latent_code, xyz[xyz_idx])
            pred_sdf = pred[:,0]
            pred_rgb = pred[:,1:]

            # compute loss
            loss_sdf, loss_rgb, loss_kl = compute_loss(pred_sdf, pred_rgb, sdf_gt, rgb_gt, lat_code_mu, lat_code_log_std, threshold_precision, param)
            loss_total = loss_sdf + loss_rgb + loss_kl

            #update weights
            loss_total.backward()
            # optimizer.step()
            optimizer_decoder.step()
            optimizer_code.step()

            # estime time left
            samples_count += batch_size
            time_left = compute_time_left(time_start, samples_count, num_model_total, num_samples_per_model, epoch, param["num_epoch"])

            # print everyl X model seen
            if samples_count%(param["num_batch_between_print"] * batch_size) == 0:

                print("Epoch {} / {:.2f}% ,loss: sdf: {:.5f}, rgb: {:.5f}, reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, min/max rgb: {:.2f}/{:.2f}, code std/mu: {:.2f}/{:.2f}, time left: {} min".format(\
                    epoch, 100 * samples_count / (num_model_total * num_samples_per_model), loss_sdf, loss_rgb, loss_kl, \
                    pred_sdf.min() * resolution, pred_sdf.max() * resolution, pred_rgb.min() * 255, pred_rgb.max() * 255, \
                    (lat_code_log_std.weight.exp()).mean(), (lat_code_mu.weight).abs().mean(), (int)(time_left/60)))

                # compute l2 dist between training models and their duplicate ones
                dist_duplicate = []
                for i in range(num_model_duplicate):
                    dist_duplicate.append((lat_code_mu(idx[i].cuda()) - lat_code_mu(idx[i + num_model_training].cuda())).mean().detach().cpu())

                # compute l2 dist between random models
                dist_random = []
                for i in range(100):
                    rand = np.random.randint(num_model_training, size = 2)
                    dist_random.append((lat_code_mu(idx[rand[0]].cuda()) - lat_code_mu(idx[rand[1]].cuda())).mean().detach().cpu())


                # print(dist_duplicate)
                l2_dup = abs(np.array(dist_duplicate)).mean()
                l2_rnd = abs(np.array(dist_random)).mean()
                print(f"avrg dist between same models: {l2_dup}")
                print(f"avrg dist between diff models: {l2_rnd}")

                #log
                logs["total"].append(loss_total.detach().cpu())
                logs["sdf"].append(loss_sdf.detach().cpu())
                logs["rgb"].append(loss_rgb.detach().cpu())
                logs["reg"].append(loss_kl.detach().cpu())
                logs["l2_dup"].append(l2_dup)
                logs["l2_rand"].append(l2_rnd)


            # print(f"Time for network pass: {time.time() - time_start}")
            # time_start = time.time()

        # scheduler.step()
        scheduler_decoder.step()
        scheduler_code.step()

    print(f"Training finish in {(int)((time.time() - time_start) / 60)} min")


    ###### Saving Decoder ######
    # save decoder
    torch.save(decoder, DECODER_PATH)


    # save logs
    with open(LOGS_PATH, "wb") as file:
        pickle.dump(logs, file)
    

    # save latent code in dict
    dict_hash_2_code = dict()
    for model_hash in list_model_hash:
        dict_hash_2_code[model_hash] = lat_code_mu(dict_model_hash_2_idx[model_hash].cuda()).detach().cpu()

    with open(LATENT_CODE_PATH, "wb") as file:
        pickle.dump(dict_hash_2_code, file)


    # save param used
    with open(PARAM_SAVE_FILE, 'w') as file:
        yaml.dump(param_all, file)


 
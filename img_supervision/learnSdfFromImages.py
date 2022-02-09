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
from dataLoader import DatasetLearnSDF

from utils import *

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
PLOT_PATH = "../../img_supervision/plots/decoder/learning_sdf/"





def init_weights(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_opt_sched(decoder, param):
    """ initialize optimizer and scheduler"""

    optimizer_decoder = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": param["eta_decoder"],
            },
        ]
    )
    

    scheduler_decoder = torch.optim.lr_scheduler.ExponentialLR(optimizer_decoder, gamma=param["gamma_decoder_LR"])

    return optimizer_decoder, scheduler_decoder




if __name__ == '__main__':
    print("Loading parameters...")

    # load parameters
    param_all = yaml.safe_load(open(PARAM_FILE))
    param_sdf = param_all["decoder_learning_sdf"]
    

    decoder_sdf = Decoder(param_all["latent_size"], "sdf", batch_norm=True).cuda()
    decoder_sdf.apply(init_weights)
    decoder_sdf.train()

    decoder_rgb = Decoder(param_all["latent_size"], "rgb", batch_norm=True).cuda()
    decoder_rgb.apply(init_weights)
    decoder_rgb.eval()

    latent_code = torch.zeros(param_all["latent_size"]).cuda()

    # initialize optimizer and scheduler
    optimizer_decoder, scheduler_decoder = init_opt_sched(decoder_sdf, param_sdf["optimizer"])


    # get models' hashs
    list_model_hash = []
    for val in glob.glob(SDF_DIR + "*.h5"):
        list_model_hash.append(os.path.basename(val).split('.')[0])

    annotations = pickle.load(open(ANNOTATIONS_PATH, 'rb'))
    num_images_per_model = len(annotations[list_model_hash[0]])


    ######################################## only used for testing ########################################
    num_images_per_model = 100
    list_model_hash = list_model_hash[0:1]
    list_model_hash = np.repeat(list_model_hash,10)
    ######################################## only used for testing ########################################

    training_dataset = DatasetLearnSDF(list_model_hash, annotations, num_images_per_model, IMAGES_PATH)
    training_generator = torch.utils.data.DataLoader(training_dataset, **param_sdf["dataLoader"])


    print("Start training rgb ...")

    time_start = time.time()

    logs = []

    images_count = 0
    for ground_truth_pixels, pos_init_ray, ray_marching_vector, min_step, max_step in training_generator:
        optimizer_decoder.zero_grad()

        batch_size = len(ground_truth_pixels)

        pos_init_ray = pos_init_ray.float().cuda().reshape(batch_size * 2500, 3)
        ray_marching_vector = ray_marching_vector.float().cuda().reshape(batch_size * 2500, 3)
        min_step = min_step.float().cuda().reshape(batch_size * 2500)
        max_step = max_step.float().cuda().reshape(batch_size * 2500)

        ground_truth_pixels = np.array(ground_truth_pixels)

        for i in range(batch_size):
            ground_truth_pixels[i,:50,:50] = cv2.resize(np.squeeze(ground_truth_pixels[i]), (50,50))
        
        ground_truth_pixels = ground_truth_pixels[:,:50,:50]
        ground_truth_pixels = torch.tensor(ground_truth_pixels,dtype=torch.float).cuda()
        ground_truth_pixels = ground_truth_pixels.reshape(batch_size, 50 * 50, 3)
        ground_truth_pixels = ground_truth_pixels.reshape(batch_size * 50 * 50, 3)

        pos_along_ray = get_pos_from_ray_marching(decoder_sdf, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step)
        # rendered_pixels, mask_car_pixels = render_pixels_from_pos(decoder_sdf, decoder_rgb, pos_along_ray, latent_code.unsqueeze(0).repeat([len(pos_along_ray),1]))

        mask_gt_silhouette = ground_truth_pixels.mean(1) != 1

        # IPython.embed()

        # pos_sampling = pos_init_ray + (torch.tensor(np.random.rand(10)) * (min_step + (max_step - min_step))).unsqueeze(1).mul(ray_marching_vector)
        # sdf = decoder_sdf(latent_code.unsqueeze(0).repeat([len(pos_sampling),1]), pos_sampling).squeeze()

        sdf = decoder_sdf(latent_code.unsqueeze(0).repeat([len(pos_along_ray),1]), pos_along_ray).squeeze()

        loss_sdf = sdf * (sdf > 0) * (mask_gt_silhouette == True)
        loss_sdf += - sdf * (sdf < 0) * (mask_gt_silhouette == False)
        loss_sdf = loss_sdf.mean()

        loss_sdf.backward()
        optimizer_decoder.step()
        scheduler_decoder.step()

        images_count += batch_size

        print(f"{images_count} / {len(list_model_hash) * num_images_per_model}")


    print("\n**************************************** VALIDATION ****************************************")


    decoder_sdf.eval()


    image_id = 0
    ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step = initialize_rendering_image(list_model_hash[0], image_id, annotations, IMAGES_PATH)

    ground_truth_image = np.array(ground_truth_image)

    pos_init_ray = torch.tensor(pos_init_ray).float().cuda()
    ray_marching_vector = torch.tensor(ray_marching_vector).float().cuda()
    min_step = torch.tensor(min_step).float().cuda()
    max_step = torch.tensor(max_step).float().cuda()

    pos_along_ray = get_pos_from_ray_marching(decoder_sdf, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step)

    rendered_image, mask_car = render_image_from_pos(decoder_sdf, decoder_rgb, pos_along_ray, latent_code, resolution=50, scaling_factor=1)

    rescale_ground_truth_image = cv2.resize(np.squeeze(ground_truth_image), rendered_image.shape[:2])
    rescale_ground_truth_image = torch.tensor(rescale_ground_truth_image,dtype=torch.float).cuda()
    mask_gt_silhouette = rescale_ground_truth_image.mean(2) != 1


    silhouette_comparison = torch.zeros([50,50,3]).cuda()
    silhouette_comparison[mask_car] += torch.tensor([0,0,1]).cuda()
    silhouette_comparison[mask_gt_silhouette] += torch.tensor([0,1,0]).cuda()


    plt.figure()
    plt.title(f"silhouette gt and rendering comparison")
    plt.imshow(silhouette_comparison.cpu().detach().numpy())
    plt.savefig(PLOT_PATH + f"{images_count}_diff_silhouettes.png")
    plt.close() 

    



import numpy as np
import torch
import pickle
import glob
import yaml
import time
import matplotlib.pyplot as plt

from networks import Decoder
from dataLoader import DatasetDecoderRGB
from utils import *

import IPython

# INPUT FILE
SDF_DIR = "../../img_supervision/sdf/"
IMAGES_PATH = "../../img_supervision/input_images/images/"
ANNOTATIONS_PATH = "../../img_supervision/input_images/annotations.pkl"
PARAM_FILE = "config/param.yaml"
DECODER_SDF_PATH = "models_and_codes/decoder_sdf.pth"

# SAVE FILE
DECODER_RGB_PATH = "models_and_codes/decoder_rgb.pth"
PARAM_SAVE_FILE = "config/param_decoder.yaml"
DICT_HASH_2_IDX_PATH = "models_and_codes/dict_hash_2_idx.pkl"
DICT_HASH_2_CODE_PATH = "models_and_codes/dict_hash_2_code.pkl"
LATENT_CODE_MU_PATH = "models_and_codes/latent_code_mu.pkl"
LATENT_CODE_LOG_STD_PATH = "models_and_codes/latent_code_log_std.pkl"
LIST_MODELS_PATH = "models_and_codes/list_models.pkl"
PLOT_PATH = "../../img_supervision/plots/decoder/intermediate_results/"

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


def compute_loss_reg(lat_code_mu, lat_code_log_std, lambda_kl):

    # regularization loss
    loss_kl = (-0.5 * (1 + lat_code_log_std.weight - lat_code_mu.weight.pow(2) - lat_code_log_std.weight.exp())).mean()
    loss_kl *= lambda_kl

    return loss_kl

def compute_loss_rgb(ground_truth_image, rendered_image, mask_car, lambda_rgb):
    
    loss = torch.nn.MSELoss(reduction='mean')

    loss_rgb = loss(ground_truth_image[mask_car == True], rendered_image[mask_car == True])
    loss_rgb *= lambda_rgb

    return loss_rgb



if __name__ == '__main__':
    print("Loading parameters...")

    # load parameters
    param_all = yaml.safe_load(open(PARAM_FILE))
    param_rgb = param_all["decoder_rgb"]

    list_model_hash = pickle.load(open(LIST_MODELS_PATH, 'rb'))
    lat_code_mu = pickle.load(open(LATENT_CODE_MU_PATH, 'rb'))
    lat_code_log_std = pickle.load(open(LATENT_CODE_LOG_STD_PATH, 'rb'))
    dict_model_hash_2_idx = pickle.load(open(DICT_HASH_2_IDX_PATH, 'rb'))

    decoder_sdf = torch.load(DECODER_SDF_PATH).cuda()
    decoder_sdf.eval()
    decoder_rgb = Decoder(param_all["latent_size"], "rgb", batch_norm=True).cuda()
    decoder_rgb.train()

    num_model = len(list_model_hash)

    # initialize optimizer and scheduler
    optimizer_decoder, optimizer_code, scheduler_decoder, scheduler_code = init_opt_sched(decoder_rgb, lat_code_mu, lat_code_log_std, param_rgb["optimizer"])

    annotations = pickle.load(open(ANNOTATIONS_PATH, 'rb'))
    num_images_per_model = len(annotations[list_model_hash[0]])

    ######################################## only used for testing ########################################
    # num_images_per_model = 10
    ######################################## only used for testing ########################################

    # # Init dataset and dataloader
    # training_dataset = DatasetDecoderRGB(list_model_hash, annotations, num_images_per_model, dict_model_hash_2_idx, IMAGES_PATH)

    ######################################## only used for testing ########################################
    training_dataset = DatasetDecoderRGB(list_model_hash[0:1], annotations, num_images_per_model, dict_model_hash_2_idx, IMAGES_PATH)
    ######################################## only used for testing ########################################

    training_generator = torch.utils.data.DataLoader(training_dataset, **param_rgb["dataLoader"])



    print("Start training rgb ...")

    time_start = time.time()

    for epoch in range (param_rgb["num_epoch"]):
        images_count = 0

        for model_idx, ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step in training_generator:
            optimizer_decoder.zero_grad()
            optimizer_code.zero_grad()

            batch_size = len(model_idx)


            # convert into cuda
            model_idx = model_idx.cuda()
            pos_init_ray = pos_init_ray.float().cuda()
            ray_marching_vector = ray_marching_vector.float().cuda()
            min_step = min_step.float().cuda()
            max_step = max_step.float().cuda()

            ground_truth_image = np.array(ground_truth_image)
            
            # Compute latent code 
            coeff_std = torch.empty(batch_size, param_all["latent_size"]).normal_().cuda()
            latent_code = coeff_std * lat_code_log_std(model_idx).exp() * param_rgb["lambda_variance"] + lat_code_mu(model_idx)
            

            for sample in range(batch_size):
                # rendered_image_temp, mask_car_temp = ray_marching_rendering(decoder, latent_code[sample], pos_init_ray[sample], ray_marching_vector[sample], min_step[sample], max_step[sample])
                rendered_image_temp, mask_car_temp = ray_marching_rendering(decoder_sdf, decoder_rgb, latent_code[sample], pos_init_ray[sample], ray_marching_vector[sample], min_step[sample], max_step[sample])
                
                if sample == 0:
                    rendered_image = torch.empty([batch_size] + list(rendered_image_temp.shape)).cuda()
                    rescale_ground_truth_image = np.empty([batch_size] + list(rendered_image_temp.shape))
                    mask_car = torch.empty([batch_size] + list(mask_car_temp.shape)).cuda()
                
                rendered_image[sample] = rendered_image_temp
                mask_car[sample] = mask_car_temp
                rescale_ground_truth_image[sample] = cv2.resize(ground_truth_image[sample], rendered_image.shape[1:3])

            rescale_ground_truth_image = torch.tensor(rescale_ground_truth_image,dtype=torch.float).cuda()

            loss_rgb = compute_loss_rgb(rescale_ground_truth_image, rendered_image, mask_car, param_rgb["lambda_rgb"])

            #update weights
            loss_rgb.backward()

            optimizer_decoder.step()
            optimizer_code.step()

            images_count += batch_size

            # print everyl X model seen
            if images_count%(param_rgb["num_batch_between_print"] * batch_size) == 0:
            # if epoch%(10) == 0:

                # estime time left
                time_left = compute_time_left(time_start, images_count, num_model, num_images_per_model, epoch, param_rgb["num_epoch"])

                print("Epoch {} / {:.2f}% , loss: rgb: {:.5f}, code std/mu: {:.2f}/{:.2f}, time left: {} min".format(\
                        epoch, 100 * images_count / (num_model * num_images_per_model), loss_rgb, \
                        (lat_code_log_std.weight.exp()).mean(), (lat_code_mu.weight).abs().mean(), (int)(time_left/60)))


                # mask_car = mask_car[0].cpu().numpy()
                # min_step = min_step[0].reshape(50,50).cpu().numpy()
                # min_step = cv2.resize(min_step, rendered_image.shape[1:3])

                # rendered_image[0][mask_car == False] = 1
                # rendered_image[0][min_step == 0] = 0
                        
                # plt.figure()
                # plt.title(f"result after {images_count} images seen")
                # plt.imshow(rendered_image[0].cpu().detach().numpy())
                # plt.savefig(PLOT_PATH + f"{epoch}_{images_count}.png")  
                # plt.close()   
                
                # plt.figure()
                # plt.title(f"ground after {images_count} images seen")
                # plt.imshow(rescale_ground_truth_image[0].cpu().detach().numpy())
                # plt.savefig(PLOT_PATH + f"{epoch}_{images_count}_gt.png")
                # plt.close()   


    print(f"Training rgb done in {(int)((time.time() - time_start) / 60)} min")




    # # SAVE EVERYTHING
    # torch.save(decoder_rgb, DECODER_RGB_PATH)

    # # save latent code in dict
    # dict_hash_2_code = dict()
    # for model_hash in list_model_hash:
    #     dict_hash_2_code[model_hash] = lat_code_mu(dict_model_hash_2_idx[model_hash].cuda()).detach().cpu()

    # with open(DICT_HASH_2_CODE_PATH, "wb") as file:
    #     pickle.dump(dict_hash_2_code, file)

    # with open(LIST_MODELS_PATH, "wb") as file:
    #     pickle.dump(list_model_hash, file)

    # with open(LATENT_CODE_MU_PATH, "wb") as file:
    #     pickle.dump(lat_code_mu, file)

    # with open(LATENT_CODE_LOG_STD_PATH, "wb") as file:
    #     pickle.dump(lat_code_log_std, file)

    # with open(DICT_HASH_2_IDX_PATH, "wb") as file:
    #     pickle.dump(dict_model_hash_2_idx, file)

    # # save param used
    # with open(PARAM_SAVE_FILE, 'w') as file:
    #     yaml.dump(param_all, file)

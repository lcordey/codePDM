import numpy as np
import torch
import pickle
import glob
import yaml
import time
import matplotlib.pyplot as plt

from networks import Decoder
from dataLoader import DatasetDecoderRGB, DatasetDecoderSDF
from marching_cubes_rgb import *
from utils import *

import IPython


DECODER_PATH = "models_and_codes/decoder.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
PARAM_FILE = "config/param.yaml"
PARAM_SAVE_FILE = "config/param_decoder.yaml"
LOGS_PATH = "../../img_supervision/logs/decoder/log.pkl"
SDF_DIR = "../../img_supervision/sdf/"
IMAGES_PATH = "../../img_supervision/input_images/images/"
ANNOTATIONS_PATH = "../../img_supervision/input_images/annotations.pkl"

PLOT_PATH = "../../img_supervision/plots/decoder/intermediate_results/"

LATENT_CODE_PATH_MU = "models_and_codes/latent_code_mu.pkl"
LATENT_CODE_PATH_DICT = "models_and_codes/latent_code_dict.pkl"
LATENT_CODE_PATH_HASH = "models_and_codes/latent_code_hash.pkl"


# num_model_duplicate = 20

######################################## only used for testing ########################################
num_model_duplicate = 0
######################################## only used for testing ########################################



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

def compute_loss_rgb(ground_truth_image, rendered_image, mask_car, lambda_rgb):
    
    loss_rgb = abs(ground_truth_image - rendered_image)
    loss_rgb[mask_car == False] = 0
    loss_rgb = loss_rgb.mean()
    loss_rgb *= lambda_rgb

    return loss_rgb

if __name__ == '__main__':
    print("Loading parameters...")

    # load parameters
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
    # list_model_hash = list_model_hash[:50]


    # with open(LATENT_CODE_PATH_HASH, "wb") as file:
    #     pickle.dump(list_model_hash, file)


    list_model_hash = pickle.load(open(LATENT_CODE_PATH_HASH, 'rb'))
    ######################################## only used for testing ########################################


    # time.sleep(3)

    # 1/0
    # # create duplicated models
    # list_model_hash_dup = []
    # for model_hash, i in zip(list_model_hash, range(num_model_duplicate)):
    #     list_model_hash_dup.append(model_hash + "_dup")

    num_model_training= len(list_model_hash)
    num_model_total = num_model_training + num_model_duplicate
    
    # initialize a random latent code for each models
    lat_code_mu, lat_code_log_std = init_lat_codes(num_model_total, param_all["latent_size"])

    # create a dictionary going from an hash to a corresponding index
    # idx = torch.arange(num_model_total).type(torch.LongTensor)
    # dict_model_hash_2_idx = dict()

    # # dict for training model
    # for model_hash, i in zip(list_model_hash, range(num_model_training)):
    #     dict_model_hash_2_idx[model_hash] = idx[i]

    # # dict for duplicated models
    # for model_hash_dup, i in zip(list_model_hash_dup, range(num_model_duplicate)):
    #     dict_model_hash_2_idx[model_hash_dup] = idx[i + num_model_training]

    # # load every models
    # print("Loading models...")
    # dict_gt_data = dict()
    # dict_gt_data["sdf"] = dict()

    # # load training data in dict
    # for model_hash, i in zip(list_model_hash, range(num_model_training)):
    #     if i%25 == 0:
    #         print(f"loading models: {i}/{num_model_training:3.0f}")

    #     # load sdf tensor
    #     h5f = h5py.File(SDF_DIR + model_hash + '.h5', 'r')
    #     h5f_tensor = torch.tensor(h5f["tensor"][()], dtype = torch.float)

    #     # split sdf and rgb then reshape
    #     sdf_gt = np.reshape(h5f_tensor[:,:,:,0], [num_samples_per_model])

    #     # normalize
    #     sdf_gt = sdf_gt / resolution
        
    #     # store in dict
    #     dict_gt_data["sdf"][model_hash] = sdf_gt

    # # load duplicate data in dict
    # for model_hash, model_hash_dup, i in zip(list_model_hash, list_model_hash_dup, range(num_model_duplicate)):
    #     if i%10 == 0:
    #         print(f"loading duplicate models: {i}/{num_model_duplicate:3.0f}")

    #     # store in dict
    #     dict_gt_data["sdf"][model_hash_dup] = dict_gt_data["sdf"][model_hash]

    # # Init dataset and dataloader
    # training_dataset = DatasetDecoderSDF(list_model_hash + list_model_hash_dup, dict_gt_data, num_samples_per_model, dict_model_hash_2_idx)
    # training_generator = torch.utils.data.DataLoader(training_dataset, **param["dataLoader_sdf"])

    # # initialize decoder
    # # decoder_sdf = Decoder(param_all["latent_size"], "sdf", batch_norm=True).cuda()

    # # initialize optimizer and scheduler
    # optimizer_decoder, optimizer_code, scheduler_decoder, scheduler_code = init_opt_sched(decoder_sdf, lat_code_mu, lat_code_log_std, param["optimizer_sdf"])

    # # logs
    # logs = dict()
    # logs["total"] = []
    # logs["sdf"] = []
    # logs["reg"] = []
    # logs["l2_dup"] = []
    # logs["l2_rand"] = []

    # print("Start training sdf ...")
    # decoder_sdf.train()

    # time_start = time.time()

    # for epoch in range (param["num_epoch_sdf"]):
    #     samples_count = 0
    #     for model_idx, sdf_gt, xyz_idx in training_generator:
    #         optimizer_decoder.zero_grad()
    #         optimizer_code.zero_grad()

    #         batch_size = len(model_idx)

    #         # transfer to gpu
    #         sdf_gt = sdf_gt.cuda()
    #         model_idx = model_idx.cuda()
    #         xyz_idx = xyz_idx

    #         # Compute latent code 
    #         coeff_std = torch.empty(batch_size, param_all["latent_size"]).normal_().cuda()
    #         latent_code = coeff_std * lat_code_log_std(model_idx).exp() * param["lambda_variance"] + lat_code_mu(model_idx)

    #         # get sdf from decoder
    #         pred = decoder_sdf(latent_code, xyz[xyz_idx])
    #         pred_sdf = pred[:,0]

    #         # compute loss
    #         loss_sdf = compute_loss_sdf(pred_sdf, sdf_gt, threshold_precision, param["lambda_sdf"])
    #         loss_kl = compute_loss_reg(lat_code_mu, lat_code_log_std, param["lambda_kl"])
    #         loss_total = loss_sdf + loss_kl

    #         #update weights
    #         loss_total.backward()
    #         # optimizer.step()
    #         optimizer_decoder.step()
    #         optimizer_code.step()

    #         samples_count += batch_size

    #         # print everyl X model seen
    #         if samples_count%(param["num_batch_between_print"] * batch_size) == 0:

    #             # estime time left
    #             time_left = compute_time_left(time_start, samples_count, num_model_total, num_samples_per_model, epoch, param["num_epoch_sdf"])

    #             print("Epoch {} / {:.2f}% ,loss: sdf: {:.5f}, reg: {:.5f}, min/max sdf: {:.2f}/{:.2f}, code std/mu: {:.2f}/{:.2f}, time left: {} min".format(\
    #                 epoch, 100 * samples_count / (num_model_total * num_samples_per_model), loss_sdf, loss_kl, \
    #                 pred_sdf.min() * resolution, pred_sdf.max() * resolution, \
    #                 (lat_code_log_std.weight.exp()).mean(), (lat_code_mu.weight).abs().mean(), (int)(time_left/60)))

    #             # compute l2 dist between training models and their duplicate ones
    #             dist_duplicate = []
    #             for i in range(num_model_duplicate):
    #                 dist_duplicate.append((lat_code_mu(idx[i].cuda()) - lat_code_mu(idx[i + num_model_training].cuda())).norm().detach().cpu())

    #             # compute l2 dist between random models
    #             dist_random = []
    #             for i in range(100):
    #                 rand = np.random.randint(num_model_training, size = 2)
    #                 dist_random.append((lat_code_mu(idx[rand[0]].cuda()) - lat_code_mu(idx[rand[1]].cuda())).norm().detach().cpu())


    #             # print(dist_duplicate)
    #             l2_dup = abs(np.array(dist_duplicate)).mean()
    #             l2_rnd = abs(np.array(dist_random)).mean()
    #             # print(f"avrg dist between same models: {l2_dup}")
    #             # print(f"avrg dist between diff models: {l2_rnd}")

    #             #log
    #             logs["total"].append(loss_total.detach().cpu())
    #             logs["sdf"].append(loss_sdf.detach().cpu())
    #             logs["reg"].append(loss_kl.detach().cpu())
    #             logs["l2_dup"].append(l2_dup)
    #             logs["l2_rand"].append(l2_rnd)
                
                
    #     scheduler_decoder.step()
    #     scheduler_code.step()

    # print(f"Training sdf done in {(int)((time.time() - time_start) / 60)} min")


    # torch.save(decoder_sdf, DECODER_PATH)

    # # save latent code in dict
    # dict_hash_2_code = dict()
    # for model_hash in list_model_hash:
    #     dict_hash_2_code[model_hash] = lat_code_mu(dict_model_hash_2_idx[model_hash].cuda()).detach().cpu()

    # with open(LATENT_CODE_PATH, "wb") as file:
    #     pickle.dump(dict_hash_2_code, file)

    # with open(LATENT_CODE_PATH_MU, "wb") as file:
    #     pickle.dump(lat_code_mu, file)

    # with open(LATENT_CODE_PATH_DICT, "wb") as file:
    #     pickle.dump(dict_model_hash_2_idx, file)


    
    decoder_sdf = torch.load(DECODER_PATH).cuda()
    lat_code_mu = pickle.load(open(LATENT_CODE_PATH_MU, 'rb'))
    dict_model_hash_2_idx = pickle.load(open(LATENT_CODE_PATH_DICT, 'rb'))

    decoder_rgb = Decoder(param_all["latent_size"], "rgb", batch_norm=True).cuda()

    torch.save(decoder_rgb, DECODER_PATH + "rgb")

    # initialize optimizer and scheduler
    optimizer_decoder, optimizer_code, scheduler_decoder, scheduler_code = init_opt_sched(decoder_rgb, lat_code_mu, lat_code_log_std, param["optimizer_rgb"])

    annotations = pickle.load(open(ANNOTATIONS_PATH, 'rb'))
    num_images_per_model = len(annotations[list_model_hash[0]])

    # Init dataset and dataloader
    # training_dataset = DatasetDecoderRGB(list_model_hash + list_model_hash_dup, annotations, num_images_per_model, dict_model_hash_2_idx, IMAGES_PATH)
    training_dataset = DatasetDecoderRGB(list_model_hash, annotations, num_images_per_model, dict_model_hash_2_idx, IMAGES_PATH)
    training_generator = torch.utils.data.DataLoader(training_dataset, **param["dataLoader_rgb"])

    # logs["rgb"] = []

    print("Start training rgb ...")
    decoder_rgb.train()
    decoder_sdf.eval()
    
    time_start = time.time()

    for epoch in range (param["num_epoch_rgb"]):
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
            latent_code = coeff_std * lat_code_log_std(model_idx).exp() * param["lambda_variance"] + lat_code_mu(model_idx)
            
            

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

            loss_rgb = compute_loss_rgb(rescale_ground_truth_image, rendered_image, mask_car, param["lambda_rgb"])

            #update weights
            loss_rgb.backward()

            optimizer_decoder.step()
            optimizer_code.step()

            images_count += batch_size

            # print everyl X model seen
            if images_count%(param["num_batch_between_print"] * batch_size) == 0:

                # estime time left
                time_left = compute_time_left(time_start, images_count, num_model_total, num_images_per_model, epoch, param["num_epoch_rgb"])

                print("Epoch {} / {:.2f}% , loss: rgb: {:.5f}, code std/mu: {:.2f}/{:.2f}, time left: {} min".format(\
                        epoch, 100 * images_count / (num_model_total * num_images_per_model), loss_rgb, \
                        (lat_code_log_std.weight.exp()).mean(), (lat_code_mu.weight).abs().mean(), (int)(time_left/60)))


                mask_car = mask_car[0].cpu().numpy()
                min_step = min_step[0].reshape(50,50).cpu().numpy()
                min_step = cv2.resize(min_step, rendered_image.shape[1:3])

                rendered_image[0][mask_car == False] = 1
                rendered_image[0][min_step == 0] = 0
                        
                # plt.figure()
                # plt.title(f"result after {images_count} images seen")
                # plt.imshow(rendered_image[0].cpu().detach().numpy())
                # plt.savefig(PLOT_PATH + f"{images_count}.png")     
                
                # plt.figure()
                # plt.title(f"ground after {images_count} images seen")
                # plt.imshow(rescale_ground_truth_image[0].cpu().detach().numpy())
                # plt.savefig(PLOT_PATH + f"{images_count}_gt.png")


    print(f"Training rgb done in {(int)((time.time() - time_start) / 60)} min")


    ###### Saving Decoder ######
    # save decoder
    # torch.save(decoder, DECODER_PATH)
    torch.save(decoder_rgb, DECODER_PATH + "rgb")


    # save logs
    # with open(LOGS_PATH, "wb") as file:
    #     pickle.dump(logs, file)
    

    # # save latent code in dict
    # dict_hash_2_code = dict()
    # for model_hash in list_model_hash:
    #     dict_hash_2_code[model_hash] = lat_code_mu(dict_model_hash_2_idx[model_hash].cuda()).detach().cpu()

    # with open(LATENT_CODE_PATH, "wb") as file:
    #     pickle.dump(dict_hash_2_code, file)


    # # save param used
    # with open(PARAM_SAVE_FILE, 'w') as file:
    #     yaml.dump(param_all, file)


 
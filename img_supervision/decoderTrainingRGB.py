from cv2 import absdiff
import numpy as np
from numpy.core.numeric import count_nonzero
from numpy.random.mtrand import sample
import torch
import pickle
import glob
import yaml
import time
import matplotlib.pyplot as plt

from networks import Decoder
from dataLoader import DatasetDecoderSDF, DatasetDecoderTrainingRGB, DatasetDecoderValidationRGB
from utils import *

import h5py



from marching_cubes_rgb import *

import IPython

# INPUT FILE
SDF_DIR = "../../img_supervision/sdf/"
IMAGES_PATH = "../../img_supervision/input_images/images/"
ANNOTATIONS_PATH = "../../img_supervision/input_images/annotations.pkl"
# IMAGES_PATH = "../../img_supervision/input_images_validation/images/"
# ANNOTATIONS_PATH = "../../img_supervision/input_images_validation/annotations.pkl"
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


def init_weights(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


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
    loss_rgb = loss(ground_truth_image[mask_car], rendered_image[mask_car])

    loss_rgb *= lambda_rgb

    return loss_rgb




def init_xyz(resolution):
    """ fill 3d grid representing 3d location to give as input to the decoder """
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz





def compute_loss(pred_sdf, pred_rgb, sdf_gt, rgb_gt, lat_code_mu, lat_code_log_std, threshold_precision, param):
    """ compute sdf, rgb and regression loss """

    loss = torch.nn.MSELoss(reduction='none')

    # assign weight of 0 for easy samples that are well trained
    weight_sdf = ~((pred_sdf > threshold_precision).squeeze() * (sdf_gt > threshold_precision).squeeze()) \
        * ~((pred_sdf < -threshold_precision).squeeze() * (sdf_gt < -threshold_precision).squeeze())

    # loss rgb
    loss_rgb = loss(pred_rgb, rgb_gt)
    loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).sum()/weight_sdf.count_nonzero()
    loss_rgb *= param["lambda_rgb"]
    

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
    decoder_rgb.apply(init_weights)
    decoder_rgb.train()

    num_model = len(list_model_hash)

    # initialize optimizer and scheduler
    optimizer_decoder, optimizer_code, scheduler_decoder, scheduler_code = init_opt_sched(decoder_rgb, lat_code_mu, lat_code_log_std, param_rgb["optimizer"])

    annotations = pickle.load(open(ANNOTATIONS_PATH, 'rb'))
    num_images_per_model = len(annotations[list_model_hash[0]])

    ######################################## only used for testing ########################################
    num_images_per_model = 1
    list_model_hash = list_model_hash[0:1]
    list_model_hash = np.repeat(list_model_hash,1000)
    ######################################## only used for testing ########################################


    training_dataset = DatasetDecoderTrainingRGB(list_model_hash, annotations, num_images_per_model, param_rgb["num_sample_per_image"], dict_model_hash_2_idx, IMAGES_PATH)
    # training_dataset = DatasetDecoderValidationRGB(np.repeat(list_model_hash,1000), annotations, num_images_per_model, dict_model_hash_2_idx, IMAGES_PATH)
    training_generator = torch.utils.data.DataLoader(training_dataset, **param_rgb["dataLoaderTraining"])

    num_images_per_model_validation = 3
    validation_dataset = DatasetDecoderValidationRGB(list_model_hash, annotations, num_images_per_model_validation, dict_model_hash_2_idx, IMAGES_PATH)
    validation_generator = torch.utils.data.DataLoader(validation_dataset, **param_rgb["dataLoaderValidation"])







    # # fill a xyz grid to give as input to the decoder 
    # xyz = init_xyz(64)

    # # test for training with 3D supervision
    # model_hash = "468780ef4ace9a422e877e82c90c24d"
    # num_samples_per_model = 64 * 64 * 64
    # dict_gt_data = dict()
    # dict_gt_data["sdf"] = dict()
    # dict_gt_data["rgb"] = dict()


    # # load sdf tensor
    # h5f = h5py.File(SDF_DIR + model_hash + '.h5', 'r')
    # h5f_tensor = torch.tensor(h5f["tensor"][()], dtype = torch.float)

    # # split sdf and rgb then reshape
    # sdf_gt = np.reshape(h5f_tensor[:,:,:,0], [num_samples_per_model])
    # rgb_gt = np.reshape(h5f_tensor[:,:,:,1:], [num_samples_per_model , 3])

    # # normalize
    # sdf_gt = sdf_gt / 64
    # rgb_gt = rgb_gt / 255


    # # store in dict
    # dict_gt_data["sdf"][model_hash] = sdf_gt
    # dict_gt_data["rgb"][model_hash] = rgb_gt


    # param_3d = param_all["param_3d"]

    # training_dataset = DatasetDecoderSDF(np.repeat(list_model_hash,20), dict_gt_data, num_samples_per_model, dict_model_hash_2_idx)
    # training_generator = torch.utils.data.DataLoader(training_dataset, **param_3d["dataLoader"])




    print("Start training rgb ...")

    time_start = time.time()

    logs = []

    for epoch in range (param_rgb["num_epoch"]):

        # for model_idx, sdf_gt, rgb_gt, xyz_idx in training_generator:
        #     optimizer_decoder.zero_grad()
        #     optimizer_code.zero_grad()

        #     batch_size = len(model_idx)

        #     # transfer to gpu
        #     sdf_gt = sdf_gt.cuda()
        #     rgb_gt = rgb_gt.cuda()
        #     model_idx = model_idx.cuda()
        #     xyz_idx = xyz_idx

        #     # Compute latent code 
        #     latent_code =  lat_code_mu(model_idx)

        #     # get sdf from decoder
        #     pred_sdf = decoder_sdf(latent_code, xyz[xyz_idx])
        #     pred_rgb = decoder_rgb(latent_code, xyz[xyz_idx])

        #     # compute loss
        #     loss_rgb = compute_loss(pred_sdf, pred_rgb, sdf_gt, rgb_gt, lat_code_mu, lat_code_log_std, 1/64, param_3d)
        #     loss_total = loss_rgb

        #     #update weights
        #     loss_total.backward()
        #     # optimizer.step()
        #     optimizer_decoder.step()
        #     optimizer_code.step()


        #     print(loss_rgb)


        print("\n**************************************** TRAINING ****************************************")


        images_count = 0
        for model_idx, ground_truth_pixels, pos_init_ray, ray_marching_vector, min_step, max_step in training_generator:
            optimizer_decoder.zero_grad()
            optimizer_code.zero_grad()

            batch_size = len(model_idx)

            # convert into cuda
            model_idx = model_idx.cuda().repeat_interleave(param_rgb["num_sample_per_image"])
            pos_init_ray = pos_init_ray.float().cuda().reshape(batch_size * param_rgb["num_sample_per_image"], 3)
            ray_marching_vector = ray_marching_vector.float().cuda().reshape(batch_size * param_rgb["num_sample_per_image"], 3)
            min_step = min_step.float().cuda().reshape(batch_size * param_rgb["num_sample_per_image"])
            max_step = max_step.float().cuda().reshape(batch_size * param_rgb["num_sample_per_image"])

            ground_truth_pixels = ground_truth_pixels.float().cuda().reshape(batch_size * param_rgb["num_sample_per_image"], 3)

            # ground_truth_pixels = np.array(ground_truth_pixels)
            # for i in range(batch_size):
            #     ground_truth_pixels[i,:50,:50] = cv2.resize(np.squeeze(ground_truth_pixels[i]), (50,50))
            
            # ground_truth_pixels = ground_truth_pixels[:,:50,:50]
            # ground_truth_pixels = torch.tensor(ground_truth_pixels,dtype=torch.float).cuda()
            # ground_truth_pixels = ground_truth_pixels.reshape(batch_size, param_rgb["num_sample_per_image"], 3)
            # ground_truth_pixels = ground_truth_pixels.reshape(batch_size * param_rgb["num_sample_per_image"], 3)
            
            # Compute latent code 
            coeff_std = torch.empty(batch_size * param_rgb["num_sample_per_image"], param_all["latent_size"]).normal_().cuda()
            latent_code = coeff_std * lat_code_log_std(model_idx).exp() * param_rgb["lambda_variance"] + lat_code_mu(model_idx)

            pos_along_ray = get_pos_from_ray_marching(decoder_sdf, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step)

            # IPython.embed()

            # ground_truth_pixels[pos_along_ray[:,2] > 0] = torch.tensor([0,0,0],dtype=torch.float).cuda()
            # ground_truth_pixels[pos_along_ray[:,2] <= 0] = torch.tensor([0.99,0.99,0.99],dtype=torch.float).cuda()

            # pos_along_ray = pos_along_ray + torch.empty_like(pos_along_ray).normal_().cuda()/64

            rendered_pixels, mask_car_pixels = render_pixels_from_pos(decoder_sdf, decoder_rgb, pos_along_ray, latent_code)

            mask_gt_silhouette = ground_truth_pixels.mean(1) != 1
            mask_loss = mask_car_pixels * mask_gt_silhouette

            loss_rgb = compute_loss_rgb(ground_truth_pixels, rendered_pixels, mask_loss, param_rgb["lambda_rgb"])

            # threshold_neg_sample = 50
            # mask_hard_samples = abs(ground_truth_pixels[mask_loss] - rendered_pixels[mask_loss]) > threshold_neg_sample/255
            # loss_rgb_neg = compute_loss_rgb(ground_truth_pixels[mask_loss], rendered_pixels[mask_loss], mask_hard_samples, param_rgb["lambda_rgb"])

            #update weights
            loss_rgb.backward()
            # loss_rgb_neg.backward()
            # loss_rgb.backward(retain_graph=True)
            # loss_rgb_neg.backward(retain_graph=True)

            optimizer_decoder.step()
            optimizer_code.step()
            scheduler_decoder.step()

            # num_neg_sample_mining = 5
            # for i in range(num_neg_sample_mining):
            #     optimizer_decoder.zero_grad()
            #     optimizer_code.zero_grad()

            #     rendered_pixels, mask_car_pixels = render_pixels_from_pos(decoder_sdf, decoder_rgb, pos_along_ray, latent_code)

            #     loss_rgb_neg = compute_loss_rgb(ground_truth_pixels[mask_loss], rendered_pixels[mask_loss], mask_hard_samples, param_rgb["lambda_rgb"])

            #     loss_rgb_neg.backward()
            #     optimizer_decoder.step()
                # optimizer_code.step()



            images_count += batch_size

            # print everyl X model seen
            if images_count%(param_rgb["num_batch_between_print"] * batch_size) == 0:

                # estime time left
                # time_left = compute_time_left(time_start, images_count, num_model, num_images_per_model, epoch, param_rgb["num_epoch"])

                # print("Epoch {} / {:.2f}% , loss: rgb: {:.5f}, code std/mu: {:.2f}/{:.2f}, time left: {} min".format(\
                #         epoch, 100 * images_count / (num_model * num_images_per_model), loss_rgb, \
                #         (lat_code_log_std.weight.exp()).mean(), (lat_code_mu.weight).abs().mean(), (int)(time_left/60)))   


                logs.append(loss_rgb.cpu().detach().numpy())
                print("Epoch {} / {} , loss: rgb: {:.5f}".format(epoch, images_count, loss_rgb))

                # print(f"{mask_hard_samples.count_nonzero()} hard sample out of {mask_loss.count_nonzero()}")





                # rendered_pixels = rendered_pixels[mask_car_pixels * mask_gt_silhouette][:100].reshape(10,10,3).cpu().detach().numpy()
                # ground_truth_pixels = ground_truth_pixels[mask_car_pixels * mask_gt_silhouette][:100].reshape(10,10,3).cpu().detach().numpy()

                # plt.figure()
                # plt.title(f"result after {images_count} images seen")
                # plt.imshow(rendered_pixels)
                # plt.savefig(PLOT_PATH + f"{epoch}_{images_count}_train_pred.png")  
                # plt.close()   

                # plt.figure()
                # plt.title(f"result after {images_count} images seen")
                # plt.imshow(ground_truth_pixels)
                # plt.savefig(PLOT_PATH + f"{epoch}_{images_count}_train_gt.png")  
                # plt.close()   

    epoch = 0

    if epoch%1 == 0:
        print("\n**************************************** VALIDATION ****************************************")
        decoder_rgb.eval()
        images_count = 0
        for model_idx, ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step in validation_generator:
            # optimizer_decoder.zero_grad()
            # optimizer_code.zero_grad()

            batch_size = len(model_idx)

            assert batch_size == 1, "batch size validation should be equal to 1"


            # convert into cuda
            model_idx = model_idx.squeeze().cuda()
            pos_init_ray = pos_init_ray.squeeze().float().cuda()
            ray_marching_vector = ray_marching_vector.squeeze().float().cuda()
            min_step = min_step.squeeze().float().cuda()
            max_step = max_step.squeeze().float().cuda()

            ground_truth_image = np.array(ground_truth_image)
            
            # Compute latent code 
            latent_code = lat_code_mu(model_idx)
            
            pos_along_ray = get_pos_from_ray_marching(decoder_sdf, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step)
            pos_along_ray = interpolate_final_pos(pos_along_ray, resolution=250, scaling_factor=1)
            rendered_image, mask_car = render_image_from_pos(decoder_sdf, decoder_rgb, pos_along_ray, latent_code, resolution=250, scaling_factor=1)
            rescale_ground_truth_image = cv2.resize(np.squeeze(ground_truth_image), rendered_image.shape[:2])
            rescale_ground_truth_image = torch.tensor(rescale_ground_truth_image,dtype=torch.float).cuda()


            mask_gt_silhouette = rescale_ground_truth_image.mean(2) != 1
            mask_loss = mask_car * mask_gt_silhouette

            loss_rgb = compute_loss_rgb(rescale_ground_truth_image, rendered_image, mask_loss, param_rgb["lambda_rgb"])

            # #update weights
            # loss_rgb.backward()

            # optimizer_decoder.step()
            # optimizer_code.step()


            images_count += batch_size

            print("Epoch {} / {} , loss: rgb: {:.5f}".format(epoch, images_count, loss_rgb))

            if epoch%10 == 0:
                mask_car = mask_car.cpu().numpy()
                # min_step = min_step.reshape(50,50).cpu().numpy()
                min_step = min_step.reshape(250,250).cpu().numpy()
                min_step = cv2.resize(min_step, rendered_image.shape[0:2])

                rendered_image[mask_car == False] = 1
                rendered_image[min_step == 0] = 1

                # image_loss = torch.zeros([250,250,3]).cuda()
                # image_loss[mask_loss] = abs(rendered_image[mask_loss] - rescale_ground_truth_image[mask_loss])

                silhouette_comparison = torch.zeros([250,250,3]).cuda()
                silhouette_comparison[mask_car] += torch.tensor([0,0,1]).cuda()
                silhouette_comparison[mask_gt_silhouette] += torch.tensor([0,1,0]).cuda()


                # print(image_loss.mean().item())
                        
                plt.figure()
                plt.title(f"result after {images_count} images seen")
                plt.imshow(rendered_image.cpu().detach().numpy())
                plt.savefig(PLOT_PATH + f"{epoch}_{images_count}_valid_pred.png")  
                plt.close()   
                
                plt.figure()
                plt.title(f"ground after {images_count} images seen")
                plt.imshow(rescale_ground_truth_image.cpu().detach().numpy())
                plt.savefig(PLOT_PATH + f"{epoch}_{images_count}_gt.png")
                plt.close() 

                plt.figure()
                plt.title(f"silhouette gt and rendering comparison")
                plt.imshow(silhouette_comparison.cpu().detach().numpy())
                plt.savefig(PLOT_PATH + f"{epoch}_{images_count}_diff_silhouettes.png")
                plt.close() 


                # plt.figure()
                # plt.title(f"ground after {images_count} images seen")
                # plt.imshow(image_loss.mean(2).cpu().detach().numpy())
                # plt.savefig(PLOT_PATH + f"{epoch}_{images_count}_diff.png")
                # plt.close() 
            
        decoder_rgb.train()


    print(f"Training rgb done in {(int)((time.time() - time_start) / 60)} min")


plt.figure()
plt.semilogy(logs)
plt.savefig(PLOT_PATH + f"_loss.png")
plt.close()


decoder_rgb.eval()


def init_xyz(resolution):
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz


resolution = 64
xyz = init_xyz(resolution)

for model_idx in range(len(list_model_hash)):
    latent_code = lat_code_mu(torch.tensor(model_idx).cuda())

    # variable to store results
    sdf_result = np.empty([resolution, resolution, resolution, 4])

    # loop because it requires too much GPU memory on my computer
    for x in range(resolution):
        # latent_code = dict_hash_2_code[model_hash].repeat(resolution * resolution, 1).cuda()
        xyz_sub_sample = xyz[x * resolution * resolution: (x+1) * resolution * resolution]
        sdf_sub_result = torch.empty([resolution * resolution, 4])

        sdf_pred = decoder_sdf(latent_code.repeat(resolution * resolution, 1), xyz_sub_sample).detach().cpu()
        sdf_pred = sdf_pred * resolution
        color_pred = decoder_rgb(latent_code.repeat(resolution * resolution, 1), xyz_sub_sample).detach().cpu()
        color_pred = torch.clamp(color_pred, 0, 1)
        color_pred = color_pred * 255

        sdf_sub_result[:,0] = sdf_pred.squeeze()
        sdf_sub_result[:,1:] = color_pred

        sdf_result[x, :, :, :] = np.reshape(sdf_sub_result[:,:], [resolution, resolution, 4])

    if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
        vertices_pred, faces_pred = marching_cubes(sdf_result[:,:,:,0])
        colors_v_pred = exctract_colors_v(vertices_pred, sdf_result)
        colors_f_pred = exctract_colors_f(colors_v_pred, faces_pred)
        off_file = "%s/_pred_%d.off" %(PLOT_PATH, model_idx)
        write_off(off_file, vertices_pred, faces_pred, colors_f_pred)
        print(f"Wrote %_pred_{model_idx}.off")
    else:
        print("surface level: 0, should be comprise in between the minimum and maximum value")



    # SAVE EVERYTHING
    torch.save(decoder_rgb, DECODER_RGB_PATH)

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

    # save param used
    with open(PARAM_SAVE_FILE, 'w') as file:
        yaml.dump(param_all, file)



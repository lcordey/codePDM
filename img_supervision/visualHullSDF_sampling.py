import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import glob
import yaml
import time
import os

from utils import *
from marching_cubes_rgb import *
from scipy import ndimage
from networks import Decoder
from dataLoader import DatasetDecoderSDF2

import IPython

MODEL_ID = 0
MAX_IMAGES = 20
RESOLUTION = 64

# INPUT FILE
SDF_DIR = "../../img_supervision/sdf/"


# IMAGES_PATH = "../../img_supervision/input_images/images/"
# ANNOTATIONS_PATH = "../../img_supervision/input_images/annotations.pkl"
IMAGES_PATH = "../../img_supervision/input_images_validation/images/"
ANNOTATIONS_PATH = "../../img_supervision/input_images_validation/annotations.pkl"
PARAM_FILE = "config/param.yaml"

# SAVE FILE
PLOT_PATH = "../../img_supervision/plots/decoder/visuall_hull_SDF/"
DECODER_SDF_PATH = "models_and_codes/decoder_sdf.pth"



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

def init_xyz(resolution):
    """ fill 3d grid representing 3d location to give as input to the decoder """
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz

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
    
def compute_time_left(time_start, samples_count, num_samples, epoch, num_epoch):
    """ Compute time left until the end of training """
    time_passed = time.time() - time_start
    num_samples_seen = epoch * num_samples + samples_count
    time_per_sample = time_passed/num_samples_seen
    estimate_total_time = time_per_sample * num_epoch * num_samples
    estimate_time_left = estimate_total_time - time_passed

    return estimate_time_left


if __name__ == '__main__':
    print("Loading parameters...")

    # load parameters
    param_all = yaml.safe_load(open(PARAM_FILE))
    param_sdf = param_all["decoder_learning_sdf"]

    # get models' hashs
    list_model_hash = []
    for val in glob.glob(SDF_DIR + "*.h5"):
        list_model_hash.append(os.path.basename(val).split('.')[0])

    annotations = pickle.load(open(ANNOTATIONS_PATH, 'rb'))
    num_images_per_model = len(annotations[list_model_hash[0]])
    num_images = min(num_images_per_model, MAX_IMAGES)

    matrix_world_to_camera = annotations["matrix_world_to_camera"]
    matrix_world_to_camera = matrix_world_to_camera[[1,0,2,3]]
    
    model_id = MODEL_ID
    model_hash = list_model_hash[model_id]
    model_annotations = annotations[model_hash].copy()
    frame = model_annotations[0]["frame"].copy()

    resolution = RESOLUTION
    step = 1/resolution
    threshold_precision = 1/resolution
    num_samples = resolution * resolution * resolution


    print("Loading sihlouettes...")
    for image_id in range(num_images_per_model):
        image_pth = IMAGES_PATH + model_hash + '/' + str(image_id) + '.png'
        input_im = imageio.imread(image_pth)

        mask_car = input_im.mean(2) == 255
        outside_dist = ndimage.distance_transform_edt(mask_car)
        inside_dist = - (ndimage.distance_transform_edt(1 - mask_car) -1)

        model_annotations[image_id]["2d_sdf"] = (outside_dist + inside_dist) / 300


    print("Computing ground truth SDF...")

    sdf_gt_estimation = np.empty([resolution, resolution, resolution, 4])
    sdf_gt_estimation[:,:,:,0] = -1e42
    sdf_gt_estimation[:,:,:,1:] = 1

    for image_id in range(num_images):

        print(f"image: {image_id} / {num_images}")

        matrix_object_to_world = model_annotations[image_id]['matrix_object_to_world'].copy()
        matrix_object_to_world = matrix_object_to_world[:,[1,0,2,3]][[2,1,0,3]]

        sdf_estimation_from_image = np.empty([resolution, resolution, resolution])

        for x in range(resolution):
            for y in range(resolution):
                for z in range(resolution):
                    pos_3d_ojbect = np.array([-0.5 + x * step, -0.5 + y * step, -0.5 + z * step])
                    pos_3d_world = matrix_object_to_world[:3,:3].dot(pos_3d_ojbect)
                    pos_2d = convert_w2c(matrix_world_to_camera, frame, pos_3d_world)
                    pos_2d[:2] *= 300

                    try:
                        img_dist = model_annotations[image_id]["2d_sdf"][pos_2d[0].astype(int), pos_2d[1].astype(int)]
                    except:
                        img_dist = 1e-10

                    sdf_estimation_from_image[x,y,z] = img_dist * pos_2d[2]

        sdf_gt_estimation[:,:,:,0] = np.maximum(sdf_gt_estimation[:,:,:,0], sdf_estimation_from_image)


    if(np.min(sdf_gt_estimation[:,:,:,0]) < 0 and np.max(sdf_gt_estimation[:,:,:,0]) > 0):
        vertices_pred, faces_pred = marching_cubes(sdf_gt_estimation[:,:,:,0])
        colors_v_pred = exctract_colors_v(vertices_pred, sdf_gt_estimation)
        colors_f_pred = exctract_colors_f(colors_v_pred, faces_pred)
        off_file = "%s/pred_%d_%d.off" %(PLOT_PATH, resolution, num_images)
        write_off(off_file, vertices_pred, faces_pred, colors_f_pred)
        with open(PLOT_PATH + f"pred_{resolution}_{num_images}.pkl", "wb") as file:
            pickle.dump(sdf_gt_estimation, file)
        print(f"Wrote pred_{resolution}_{num_images}.off")
    else:
        print("surface level: 0, should be comprise in between the minimum and maximum value")


    sdf_gt_estimation = sdf_gt_estimation[:,:,:,0].squeeze().astype(np.float32)
    sdf_gt_estimation = sdf_gt_estimation.reshape(resolution * resolution * resolution)

    latent_code = torch.zeros(param_all["latent_size"]).cuda()

    # fill a xyz grid to give as input to the decoder 
    xyz = init_xyz(resolution)

    decoder_sdf = Decoder(param_all["latent_size"], "sdf", batch_norm=True).cuda()
    decoder_sdf.apply(init_weights)
    decoder_sdf.train()

    decoder_rgb = Decoder(param_all["latent_size"], "rgb", batch_norm=True).cuda()
    decoder_rgb.apply(init_weights)
    decoder_rgb.train()

    optimizer_decoder, scheduler_decoder = init_opt_sched(decoder_sdf, param_sdf["optimizer"])

    # Init dataset and dataloader
    training_dataset = DatasetDecoderSDF2(sdf_gt_estimation)
    training_generator = torch.utils.data.DataLoader(training_dataset, **param_sdf["dataLoader"])


    time_start = time.time()

    for epoch in range(param_sdf["num_epoch"]):

        print(f"\nepoch: {epoch}")

        samples_count = 0
        for sdf_gt, xyz_idx in training_generator:
            optimizer_decoder.zero_grad()

            batch_size = len(sdf_gt)
            sdf_gt = sdf_gt.cuda()

            pred_sdf = decoder_sdf(latent_code.unsqueeze(0).repeat(batch_size, 1), xyz[xyz_idx])
            loss_sdf = compute_loss_sdf(pred_sdf, sdf_gt, threshold_precision, param_sdf["lambda_sdf"])

            #update weights
            loss_sdf.backward()
            optimizer_decoder.step()
            samples_count += batch_size

            time_left = compute_time_left(time_start, samples_count, num_samples, epoch, param_sdf["num_epoch"])

        print(f"loss_sdf: {loss_sdf}")
        print(f"time left: {time_left}")

        scheduler_decoder.step()


    print("\nevaluate result...")

    # variable to store results
    sdf_result = np.empty([resolution, resolution, resolution, 4])
    pred_all = np.empty([resolution * resolution, 4])

    decoder_sdf.eval()

    # loop because it requires too much GPU memory on my computer
    for x in range(resolution):
        xyz_sub_sample = xyz[x * resolution * resolution: (x+1) * resolution * resolution]

        pred_sdf = decoder_sdf(latent_code.unsqueeze(0).repeat(resolution * resolution, 1), xyz_sub_sample).detach().cpu()
        pred_sdf = pred_sdf * resolution

        pred_rgb = decoder_rgb(latent_code.unsqueeze(0).repeat(resolution * resolution, 1), xyz_sub_sample).detach().cpu()
        pred_rgb = torch.clamp(pred_rgb, 0, 1)
        pred_rgb = pred_rgb * 255

        pred_all[:,0] = pred_sdf.squeeze()
        pred_all[:,1:] = pred_rgb

        sdf_result[x, :, :, :] = np.reshape(pred_all, [resolution, resolution, 4])


    if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
        vertices_pred, faces_pred = marching_cubes(sdf_result[:,:,:,0])
        colors_v_pred = exctract_colors_v(vertices_pred, sdf_result)
        colors_f_pred = exctract_colors_f(colors_v_pred, faces_pred)
        off_file = "%s/pred_from_sdf.off" %(PLOT_PATH)
        write_off(off_file, vertices_pred, faces_pred, colors_f_pred)
        print("Wrote pred_from_sdf.off")
    else:
        print("surface level: 0, should be comprise in between the minimum and maximum value")

    # SAVE EVERYTHING
    torch.save(decoder_sdf, DECODER_SDF_PATH)



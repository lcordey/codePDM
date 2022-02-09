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
from dataLoader import DatasetDecoderTrainingRGB_2, DatasetDecoderValidationRGB_2
from utils import *

from marching_cubes_rgb import *

import IPython

MODEL_ID = 0


# INPUT FILE
SDF_DIR = "../../img_supervision/sdf/"
# IMAGES_PATH = "../../img_supervision/input_images/images/"
# ANNOTATIONS_PATH = "../../img_supervision/input_images/annotations.pkl"
IMAGES_PATH = "../../img_supervision/input_images_validation/images/"
ANNOTATIONS_PATH = "../../img_supervision/input_images_validation/annotations.pkl"
PARAM_FILE = "config/param.yaml"
DECODER_SDF_PATH = "models_and_codes/decoder_sdf.pth"

# SAVE FILE
DECODER_RGB_PATH = "models_and_codes/decoder_rgb.pth"
PARAM_SAVE_FILE = "config/param_decoder.yaml"
PLOT_PATH = "../../img_supervision/plots/decoder/visual_hull_RGB/"




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


def compute_time_left(time_start, samples_count, num_samples):
    """ Compute time left until the end of training """
    time_passed = time.time() - time_start
    num_samples_seen = samples_count
    time_per_sample = time_passed/num_samples_seen
    estimate_total_time = time_per_sample * num_samples
    estimate_time_left = estimate_total_time - time_passed

    return estimate_time_left


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




if __name__ == '__main__':
    print("Loading parameters...")

    # load parameters
    param_all = yaml.safe_load(open(PARAM_FILE))
    param_rgb = param_all["decoder_learning_rgb"]
    
    # get models' hashs
    list_model_hash = []
    for val in glob.glob(SDF_DIR + "*.h5"):
        list_model_hash.append(os.path.basename(val).split('.')[0])

    annotations = pickle.load(open(ANNOTATIONS_PATH, 'rb'))
    num_images_per_model = len(annotations[list_model_hash[0]])

    model_id = MODEL_ID
    model_hash = list_model_hash[model_id]

    decoder_sdf = torch.load(DECODER_SDF_PATH).cuda()
    decoder_sdf.eval()

    decoder_rgb = Decoder(param_all["latent_size"], "rgb", batch_norm=True).cuda()
    decoder_rgb.apply(init_weights)
    decoder_rgb.train()

    optimizer_decoder, scheduler_decoder = init_opt_sched(decoder_rgb, param_rgb["optimizer"])

    num_images_per_model = 3

    # Init dataset and dataloader
    training_dataset = DatasetDecoderTrainingRGB_2(model_hash, param_rgb["num_repetition_image"], num_images_per_model, param_rgb["num_sample_per_image"], annotations, IMAGES_PATH)
    training_generator = torch.utils.data.DataLoader(training_dataset, **param_rgb["dataLoaderTraining"])

    validation_dataset = DatasetDecoderValidationRGB_2(model_hash, param_rgb["num_images_validation"], annotations, IMAGES_PATH)
    validation_generator = torch.utils.data.DataLoader(validation_dataset, **param_rgb["dataLoaderValidation"])


    print("Start training rgb ...")

    time_start = time.time()
    logs = []

    images_count = 0
    for ground_truth_pixels, pos_init_ray, ray_marching_vector, min_step, max_step in training_generator:
        optimizer_decoder.zero_grad()

        batch_size = len(ground_truth_pixels)

        # convert into cuda
        pos_init_ray = pos_init_ray.float().cuda().reshape(batch_size * param_rgb["num_sample_per_image"], 3)
        ray_marching_vector = ray_marching_vector.float().cuda().reshape(batch_size * param_rgb["num_sample_per_image"], 3)
        min_step = min_step.float().cuda().reshape(batch_size * param_rgb["num_sample_per_image"])
        max_step = max_step.float().cuda().reshape(batch_size * param_rgb["num_sample_per_image"])

        ground_truth_pixels = ground_truth_pixels.float().cuda().reshape(batch_size * param_rgb["num_sample_per_image"], 3)

        latent_code = torch.zeros(param_all["latent_size"]).cuda().unsqueeze(0).repeat(batch_size * param_rgb["num_sample_per_image"], 1)

        pos_along_ray = get_pos_from_ray_marching(decoder_sdf, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step)
        rendered_pixels, mask_car_pixels = render_pixels_from_pos(decoder_sdf, decoder_rgb, pos_along_ray, latent_code)

        mask_gt_silhouette = ground_truth_pixels.mean(1) != 1
        mask_loss = mask_car_pixels * mask_gt_silhouette

        loss_rgb = compute_loss_rgb(ground_truth_pixels, rendered_pixels, mask_loss, param_rgb["lambda_rgb"])

        #update weights
        loss_rgb.backward()
        optimizer_decoder.step()
        scheduler_decoder.step()

        images_count += batch_size

        # print everyl X model seen
        if images_count%(param_rgb["num_batch_between_print"] * batch_size) == 0:

            time_left = compute_time_left(time_start, images_count, num_images_per_model * param_rgb["num_repetition_image"])
            

            print("Epoch {} / {} , loss rgb: {:.5f}, time left: {} sec".format(images_count, num_images_per_model * param_rgb["num_repetition_image"], loss_rgb, time_left))
            logs.append(loss_rgb.cpu().detach().numpy())


    print(f"Training rgb done in {(int)((time.time() - time_start) / 60)} min")

    print("\n**************************************** VALIDATION ****************************************")
    decoder_rgb.eval()
    images_count = 0

    for ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step in validation_generator:

        batch_size = len(ground_truth_image)
        assert batch_size == 1, "batch size validation should be equal to 1"

        # convert into cuda
        pos_init_ray = pos_init_ray.squeeze().float().cuda()
        ray_marching_vector = ray_marching_vector.squeeze().float().cuda()
        min_step = min_step.squeeze().float().cuda()
        max_step = max_step.squeeze().float().cuda()

        ground_truth_image = np.array(ground_truth_image)
        
        # Compute latent code 
        latent_code = torch.zeros(param_all["latent_size"]).cuda()
        
        pos_along_ray = get_pos_from_ray_marching(decoder_sdf, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step)
        pos_along_ray = interpolate_final_pos(pos_along_ray, resolution=250, scaling_factor=1)
        rendered_image, mask_car = render_image_from_pos(decoder_sdf, decoder_rgb, pos_along_ray, latent_code, resolution=250, scaling_factor=1)
        rescale_ground_truth_image = cv2.resize(np.squeeze(ground_truth_image), rendered_image.shape[:2])
        rescale_ground_truth_image = torch.tensor(rescale_ground_truth_image,dtype=torch.float).cuda()

        mask_gt_silhouette = rescale_ground_truth_image.mean(2) != 1
        mask_loss = mask_car * mask_gt_silhouette

        loss_rgb = compute_loss_rgb(rescale_ground_truth_image, rendered_image, mask_loss, param_rgb["lambda_rgb"])

        print("loss: rgb: {:.5f}".format(loss_rgb))

        mask_car = mask_car.cpu().numpy()
        rendered_image[mask_car == False] = 1

        min_step = min_step.reshape(250,250).cpu().numpy()
        min_step = cv2.resize(min_step, rendered_image.shape[0:2])
        rendered_image[min_step == 0] = 1


        silhouette_comparison = torch.zeros([250,250,3]).cuda()
        silhouette_comparison[mask_car] += torch.tensor([1,0,0]).cuda()
        silhouette_comparison[mask_gt_silhouette] += torch.tensor([0,1,0]).cuda()

                
        plt.figure()
        plt.title(f"result after {images_count} images seen")
        plt.imshow(rendered_image.cpu().detach().numpy())
        plt.savefig(PLOT_PATH + f"{images_count}_pred.png")  
        plt.close()   
        
        plt.figure()
        plt.title(f"ground after {images_count} images seen")
        plt.imshow(rescale_ground_truth_image.cpu().detach().numpy())
        plt.savefig(PLOT_PATH + f"{images_count}_gt.png")
        plt.close() 

        plt.figure()
        plt.title(f"silhouette gt and rendering comparison")
        plt.imshow(silhouette_comparison.cpu().detach().numpy())
        plt.savefig(PLOT_PATH + f"{images_count}_diff_silhouettes.png")
        plt.close() 


        images_count += batch_size


    plt.figure()
    plt.semilogy(logs)
    plt.savefig(PLOT_PATH + f"loss.png")
    plt.close()

    print("Generate 3D model...")

    # fill a xyz grid to give as input to the decoder 
    resolution = 100
    xyz = init_xyz(resolution)
    latent_code = torch.zeros(param_all["latent_size"]).cuda()

    # variable to store results
    sdf_result = np.empty([resolution, resolution, resolution, 4])
    pred_all = np.empty([resolution * resolution, 4])

    decoder_rgb.eval()

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
        off_file = "%s/pred.off" %(PLOT_PATH)
        write_off(off_file, vertices_pred, faces_pred, colors_f_pred)
        print("Wrote pred.off")
    else:
        print("surface level: 0, should be comprise in between the minimum and maximum value")



    # SAVE EVERYTHING
    torch.save(decoder_rgb, DECODER_RGB_PATH)


    # save param used
    with open(PARAM_SAVE_FILE, 'w') as file:
        yaml.dump(param_all, file)
import torch
import pickle
import imageio
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

from skimage.transform import downscale_local_mean
from utils import *
from networks import *
from marching_cubes_rgb import *

DECODER_PATH = "../img_supervision/models_and_codes/decoder.pth"
LATENT_CODE_PATH = "../img_supervision/models_and_codes/latent_code.pkl"
ANNOTATIONS_PATH = "../../img_supervision/input_images/annotations.pkl"
IMAGES_PATH = "../../img_supervision/input_images/images/"
MATRIX_PATH = "../../img_supervision/input_images/matrix_w2c.pkl"
PLOT_PATH = "../../img_supervision/plots/decoder/intermediate_results/"



LAMBDA_RGB = 0.01


def compute_loss_rgb(ground_truth_image, rendered_image, mask_car, lambda_rgb):
    
    loss = torch.nn.MSELoss(reduction='mean')

    loss_rgb = loss(ground_truth_image[mask_car == True], rendered_image[mask_car == True])
    loss_rgb *= lambda_rgb

    return loss_rgb
    

model_id = 0
image_id = 0

annotations = pickle.load(open(ANNOTATIONS_PATH, "rb"))
dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))
decoder_sdf = torch.load(DECODER_PATH).cuda()
decoder_sdf.eval()

list_hash = list(dict_hash_2_code.keys())
model_hash = list_hash[model_id]
latent_code = dict_hash_2_code[model_hash].cuda()

decoder_rgb = Decoder(2, "rgb", batch_norm=True).cuda()
# decoder_rgb = Decoder(2, "rgb", batch_norm=False).cuda()
decoder_rgb.train()
# decoder_rgb = DecoderComplex(2, "rgb", batch_norm=True).cuda()


# latent_code.requires_grad = True




optimizer = torch.optim.Adam(
    [
        {
            # "params": latent_code,
            "params": decoder_rgb.parameters(),
            # "lr": 0.0003,
            # "lr": 0.0002,
            "lr": 0.0001,
            # "lr": 0.000005,
        },

    ]
)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)

fig, axs = plt.subplots(4,4, figsize=(10,10))

step_between_results = 5
num_images = 2

for iter in range(15 * step_between_results):
    optimizer.zero_grad()
    loss_rgb = 0
    for image_id in range(num_images):

        ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step = initialize_rendering(model_hash, image_id, annotations, IMAGES_PATH)
        ground_truth_image = np.array(ground_truth_image)
        ground_truth_image = cv2.resize(ground_truth_image, (100,100))

        ray_marching_vector = torch.tensor(ray_marching_vector,dtype=torch.float).cuda()
        pos_init_ray = torch.tensor(pos_init_ray,dtype=torch.float).cuda()
        min_step = torch.tensor(min_step,dtype=torch.float).cuda()
        max_step = torch.tensor(max_step,dtype=torch.float).cuda()

        rendered_image, mask_car = ray_marching_rendering(decoder_sdf, decoder_rgb, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step)

        loss_rgb += 10000/2 * compute_loss_rgb(torch.tensor(ground_truth_image,dtype=torch.float).cuda(), rendered_image, mask_car, LAMBDA_RGB)
        # loss_rgb = compute_loss_rgb(torch.tensor(ground_truth_image,dtype=torch.float).cuda(), rendered_image, mask_car, LAMBDA_RGB)
        # loss_rgb *= 10000

    loss_rgb.backward()
    optimizer.step()
    scheduler.step()


    if iter %step_between_results == 0:
        tmp = (int)(iter/step_between_results)
        rendered_image = rendered_image.cpu().detach().numpy()
        mask_min_step = min_step.cpu().detach().clone().numpy()
        mask_car = mask_car.cpu().numpy()
        mask_min_step = mask_min_step.reshape(50,50)
        mask_min_step = cv2.resize(mask_min_step, rendered_image.shape[0:2])

        rendered_image[mask_car == False] = 1
        rendered_image[mask_min_step == 0] = 0

        axs[(int)((tmp - tmp%4)/4), tmp%4].imshow(rendered_image)
        print(f"loss rgb: {loss_rgb.item():.5f}")


axs[3,3].imshow(ground_truth_image)

print("saving...")
fig.savefig(PLOT_PATH + "__test.png")
print("start computing 3D models...")





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
    off_file = "%s/_test.off" %(PLOT_PATH)
    write_off(off_file, vertices_pred, faces_pred, colors_f_pred)
    print("Wrote %_test.off")
else:
    print("surface level: 0, should be comprise in between the minimum and maximum value")


torch.save(decoder_rgb, PLOT_PATH + "_decoder_rgb.pth")




num_images = 8

decoder_rgb = torch.load(PLOT_PATH + "_decoder_rgb.pth").cuda()
# decoder_rgb.train()
decoder_rgb.eval()

fig, axs = plt.subplots(4,4, figsize=(10,10))
for image_id in range(num_images):
    ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step = initialize_rendering(model_hash, image_id, annotations, IMAGES_PATH)
    ground_truth_image = np.array(ground_truth_image)
    ground_truth_image = cv2.resize(ground_truth_image, (100,100))

    ray_marching_vector = torch.tensor(ray_marching_vector,dtype=torch.float).cuda()
    pos_init_ray = torch.tensor(pos_init_ray,dtype=torch.float).cuda()
    min_step = torch.tensor(min_step,dtype=torch.float).cuda()
    max_step = torch.tensor(max_step,dtype=torch.float).cuda()

    rendered_image, mask_car = ray_marching_rendering(decoder_sdf, decoder_rgb, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step)
    rendered_image = rendered_image.cpu().detach().numpy()

    col = image_id%4
    row = (int)((image_id - col) / 4 * 2)
    axs[row, col].imshow(rendered_image)
    axs[row + 1, col].imshow(ground_truth_image)


print("saving...")
fig.savefig(PLOT_PATH + "__test_2.png")


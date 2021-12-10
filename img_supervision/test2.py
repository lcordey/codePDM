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

DECODER_PATH = "../img_supervision/models_and_codes/decoder.pth"
LATENT_CODE_PATH = "../img_supervision/models_and_codes/latent_code.pkl"
ANNOTATIONS_PATH = "../../img_supervision/input_images/annotations.pkl"
IMAGES_PATH = "../../img_supervision/input_images/images/"
MATRIX_PATH = "../../img_supervision/input_images/matrix_w2c.pkl"
PLOT_PATH = "../../img_supervision/plots/decoder/intermediate_results/"

annotations = pickle.load(open(ANNOTATIONS_PATH, "rb"))
dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))
decoder_sdf = torch.load(DECODER_PATH).cuda()
decoder_rgb = torch.load(DECODER_PATH + "rgb").cuda()
decoder_sdf.eval()
decoder_rgb.train()

save = pickle.load(open(PLOT_PATH + "0_0.pkl", 'rb'))


model_hash = save["model_hash"]
ground_truth_image = save["ground_truth_image"]
pos_init_ray = save["pos_init_ray"]
ray_marching_vector = save["ray_marching_vector"]
min_step = save["min_step"]
max_step = save["max_step"]

ground_truth_image = np.array(ground_truth_image)

ray_marching_vector = torch.tensor(ray_marching_vector,dtype=torch.float).cuda()
pos_init_ray = torch.tensor(pos_init_ray,dtype=torch.float).cuda()
min_step = torch.tensor(min_step,dtype=torch.float).cuda()
max_step = torch.tensor(max_step,dtype=torch.float).cuda()

code_gt = dict_hash_2_code[model_hash].cuda()

rendered_image, mask_car = ray_marching_rendering(decoder_sdf, decoder_rgb, code_gt, pos_init_ray, ray_marching_vector, min_step, max_step)


rendered_image = rendered_image.cpu().detach().numpy()
min_step = min_step.cpu().detach().numpy()
mask_car = mask_car.cpu().numpy()
min_step = min_step.reshape(50,50)

rendered_image[mask_car == False] = 0
rendered_image[min_step == 0] = 1

plt.figure()
plt.imshow(rendered_image)
plt.savefig(PLOT_PATH + "0_0_test.png")

print(code_gt)

print("done")

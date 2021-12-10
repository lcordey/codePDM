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
# ANNOTATIONS_PATH = "../../img_supervision/input_images_validation/annotations.pkl"
ANNOTATIONS_PATH = "../../img_supervision/input_images/annotations.pkl"
# IMAGES_PATH = "../../img_supervision/input_images_validation/images/"
IMAGES_PATH = "../../img_supervision/input_images/images/"
# MATRIX_PATH = "../../img_supervision/input_images_validation/matrix_w2c.pkl"
MATRIX_PATH = "../../img_supervision/input_images/matrix_w2c.pkl"

# RAY_MARCHING_RESOLUTION = 50

# BOUND_MAX_CUBE = 0.5
# BOUND_MIN_CUBE = -0.5
# THRESHOLD = 1/64
# MAX_STEP = 1/16
# MAX_ITER = 20
# SCALING_FACTOR = 1

# def get_camera_matrix_and_frame(model_hash, image_id, annotations):
#     frame = annotations[model_hash][image_id]['frame'].copy()
#     matrix_object_to_world = annotations[model_hash][image_id]['matrix_object_to_world'].copy()
#     matrix_world_to_camera = annotations["matrix_world_to_camera"]

#     matrix_world_to_camera = matrix_world_to_camera[[1,0,2,3]]
#     matrix_object_to_world = matrix_object_to_world[:,[1,0,2,3]][[2,1,0,3]]

#     matrix_camera_to_world = np.linalg.inv(matrix_world_to_camera)
#     matrix_world_to_object = np.linalg.inv(matrix_object_to_world)

#     matrix_camera_to_object = matrix_world_to_object.dot(matrix_camera_to_world)

#     return frame, matrix_camera_to_object

# def initialize_rendering(model_hash, image_id, annotations, image_path):

#     image_pth = image_path + model_hash + '/' + str(image_id) + '.png'
#     ground_truth_image = imageio.imread(image_pth)/255

#     resolution = RAY_MARCHING_RESOLUTION

#     frame, matrix_camera_to_object = get_camera_matrix_and_frame(model_hash, image_id, annotations)

#     cam_pos_cam_coord = np.array([0,0,0,1])
#     cam_pos_obj_coord = matrix_camera_to_object.dot(cam_pos_cam_coord)

#     pos_init_ray = np.ones([resolution * resolution, 3])
#     pos_init_ray[:,0] *= cam_pos_obj_coord[0]
#     pos_init_ray[:,1] *= cam_pos_obj_coord[1]
#     pos_init_ray[:,2] *= cam_pos_obj_coord[2]

#     ray_marching_vector = np.empty([resolution * resolution, 3])

#     for i in range(resolution):
#         for j in range(resolution):

#             # select pixel to render
#             pixel_pos = np.array([i/(resolution-1),j/(resolution-1)])

#             # convert pixel in world coordinate
#             pixel_pos_cam_coord = convert_view_to_camera_coordinates(frame, pixel_pos)
#             pixel_pos_obj_coord = matrix_camera_to_object.dot(pixel_pos_cam_coord)

#             ray_marching_vector[i * resolution + j,:] = ((pixel_pos_obj_coord - cam_pos_obj_coord)/np.linalg.norm(pixel_pos_obj_coord - cam_pos_obj_coord))[:3]


#     min_step = np.zeros([resolution * resolution, 3])
#     max_step = np.ones([resolution * resolution, 3]) * 1e38

#     min_step[ray_marching_vector > 0] = (BOUND_MIN_CUBE - pos_init_ray[ray_marching_vector > 0]) / ray_marching_vector[ray_marching_vector > 0]
#     min_step[ray_marching_vector < 0] = (BOUND_MAX_CUBE - pos_init_ray[ray_marching_vector < 0]) / ray_marching_vector[ray_marching_vector < 0]

#     max_step[ray_marching_vector > 0] = (BOUND_MAX_CUBE - pos_init_ray[ray_marching_vector > 0]) / ray_marching_vector[ray_marching_vector > 0]
#     max_step[ray_marching_vector < 0] = (BOUND_MIN_CUBE - pos_init_ray[ray_marching_vector < 0]) / ray_marching_vector[ray_marching_vector < 0]

#     max_step[(ray_marching_vector == 0) * ~((pos_init_ray > BOUND_MIN_CUBE) * (pos_init_ray < BOUND_MAX_CUBE))] = 0

#     min_step = min_step.max(1)
#     max_step = max_step.min(1)

#     max_step[min_step >= max_step] = 0
#     min_step[min_step >= max_step] = 0



#     return ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step

# # def ray_marching_rendering(decoder, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step):

# #     resolution = RAY_MARCHING_RESOLUTION

# #     # convert into cuda tensor
# #     ray_marching_vector = torch.tensor(ray_marching_vector,dtype=torch.float).cuda()
# #     pos_init_ray = torch.tensor(pos_init_ray,dtype=torch.float).cuda()
# #     min_step = torch.tensor(min_step,dtype=torch.float).cuda()
# #     max_step = torch.tensor(max_step,dtype=torch.float).cuda()

# #     # marching safely until the "cube" representing the zone where the sdf was trained, and therefore contain the whole object
# #     min_pos = pos_init_ray + min_step.unsqueeze(1).mul(ray_marching_vector)
# #     pos_along_ray = min_pos
# #     sum_step = min_step

# #     ### max pos not used, function ca be improved to stop computing sdf after max pos is reached
# #     # max_pos = pos_init_ray + max_step.unsqueeze(1).mul(ray_marching_vector)

# #     # only compute ray marching for ray passing through the cube
# #     mask_cube = (min_step != 0)
# #     mask_ray_still_marching = mask_cube

# #     # define ratio to accelerate ray marching
# #     ratio = torch.ones([resolution * resolution]).cuda()
# #     sdf = torch.zeros([resolution * resolution]).cuda()

# #     for iter in range(MAX_ITER):
# #         # compute sdf values
# #         sdf[mask_ray_still_marching] = decoder(latent_code.unsqueeze(0).repeat([mask_ray_still_marching.count_nonzero(),1]), pos_along_ray[mask_ray_still_marching])[:,0].detach()

# #         # scaling: unit of 1 from the decoder's prediction correspond to a unit of 1 in the object coordinate
# #         sdf *= 1

# #         # compute the ratio using difference between old an new sdf values,
# #         if iter == 0:
# #             # store first sdf
# #             old_sdf = sdf
# #         else:
# #             # only compute ratio when the sdf is high enough and decreasing
# #             mask_ratio = (sdf > THRESHOLD) * (sdf < old_sdf)
# #             ratio[mask_ratio] = old_sdf[mask_ratio]/(old_sdf[mask_ratio] - sdf[mask_ratio])

# #             # store new sdf
# #             old_sdf[mask_ray_still_marching] = sdf[mask_ray_still_marching]
            
# #             # accelarating the step, this is only possible because the decoder tends to underestimate the distance due to the design of the loss
# #             sdf[mask_ratio] *= ratio[mask_ratio]

# #         # clamp value to prevent undesired issues
# #         sdf = sdf.clamp(None, MAX_STEP)

# #         # march along the ray
# #         step = sdf
# #         sum_step += step
# #         pos_along_ray = pos_along_ray + step.unsqueeze(1).mul(ray_marching_vector)

# #         mask_ray_still_marching = mask_ray_still_marching * (sum_step < max_step)

# #     # interpolate final position for a higher resolution
# #     pos_along_ray = pos_along_ray.reshape(resolution, resolution, 3).permute(2,0,1).unsqueeze(0)
# #     pos_along_ray = torch.nn.functional.interpolate(pos_along_ray, scale_factor = SCALING_FACTOR, mode='bilinear', align_corners=False)
# #     pos_along_ray = pos_along_ray.squeeze().permute(1,2,0).reshape(resolution * resolution * SCALING_FACTOR * SCALING_FACTOR, 3)

# #     # compute corresponding sdf
# #     sdf_and_rgb = decoder(latent_code.unsqueeze(0).repeat([resolution * resolution * SCALING_FACTOR * SCALING_FACTOR,1]), pos_along_ray)[:,:].detach()
# #     sdf_and_rgb = sdf_and_rgb.reshape(resolution * SCALING_FACTOR, resolution * SCALING_FACTOR, 4)
# #     sdf = sdf_and_rgb[:,:,0]
# #     rgb = sdf_and_rgb[:,:,1:]

# #     # init image with white background
# #     rendered_image = torch.ones([resolution * SCALING_FACTOR, resolution * SCALING_FACTOR,3]).cuda()

# #     mask_car = sdf < THRESHOLD
# #     rendered_image[mask_car] = rgb[mask_car]

# #     return rendered_image, mask_car


# def ray_marching_rendering(decoder_sdf, decoder_rgb, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step):

#     resolution = RAY_MARCHING_RESOLUTION

#     # convert into cuda tensor
#     ray_marching_vector = torch.tensor(ray_marching_vector,dtype=torch.float).cuda()
#     pos_init_ray = torch.tensor(pos_init_ray,dtype=torch.float).cuda()
#     min_step = torch.tensor(min_step,dtype=torch.float).cuda()
#     max_step = torch.tensor(max_step,dtype=torch.float).cuda()

#     # marching safely until the "cube" representing the zone where the sdf was trained, and therefore contain the whole object
#     min_pos = pos_init_ray + min_step.unsqueeze(1).mul(ray_marching_vector)
#     pos_along_ray = min_pos
#     sum_step = min_step


#     # only compute ray marching for ray passing through the cube
#     mask_cube = (min_step != 0)
#     mask_ray_still_marching = mask_cube

#     # define ratio to accelerate ray marching
#     ratio = torch.ones([resolution * resolution]).cuda()
#     sdf = torch.zeros([resolution * resolution]).cuda()

#     for iter in range(MAX_ITER):
#         # compute sdf values
#         sdf[mask_ray_still_marching] = decoder_sdf(latent_code.unsqueeze(0).repeat([mask_ray_still_marching.count_nonzero(),1]), pos_along_ray[mask_ray_still_marching])[:,0].detach()

#         # scaling: unit of 1 from the decoder's prediction correspond to a unit of 1 in the object coordinate
#         sdf *= 1

#         # compute the ratio using difference between old an new sdf values,
#         if iter == 0:
#             # store first sdf
#             old_sdf = sdf
#         else:
#             # only compute ratio when the sdf is high enough and decreasing
#             mask_ratio = (sdf > THRESHOLD) * (sdf < old_sdf)
#             ratio[mask_ratio] = old_sdf[mask_ratio]/(old_sdf[mask_ratio] - sdf[mask_ratio])

#             # store new sdf
#             old_sdf[mask_ray_still_marching] = sdf[mask_ray_still_marching]
            
#             # accelarating the step, this is only possible because the decoder tends to underestimate the distance due to the design of the loss
#             sdf[mask_ratio] *= ratio[mask_ratio]

#         # clamp value to prevent undesired issues
#         sdf = sdf.clamp(None, MAX_STEP)

#         # march along the ray
#         step = sdf
#         sum_step += step
#         pos_along_ray = pos_along_ray + step.unsqueeze(1).mul(ray_marching_vector)

#         mask_ray_still_marching = mask_ray_still_marching * (sum_step < max_step)

#     # interpolate final position for a higher resolution
#     pos_along_ray = pos_along_ray.reshape(resolution, resolution, 3).permute(2,0,1).unsqueeze(0)
#     pos_along_ray = torch.nn.functional.interpolate(pos_along_ray, scale_factor = SCALING_FACTOR, mode='bilinear', align_corners=False)
#     pos_along_ray = pos_along_ray.squeeze().permute(1,2,0).reshape(resolution * resolution * SCALING_FACTOR * SCALING_FACTOR, 3)

#     # compute corresponding sdf
#     sdf = decoder_sdf(latent_code.unsqueeze(0).repeat([resolution * resolution * SCALING_FACTOR * SCALING_FACTOR,1]), pos_along_ray)[:,:]
#     rgb = decoder_rgb(latent_code.unsqueeze(0).repeat([resolution * resolution * SCALING_FACTOR * SCALING_FACTOR,1]), pos_along_ray)[:,:]
#     sdf = sdf.reshape(resolution * SCALING_FACTOR, resolution * SCALING_FACTOR)
#     rgb = rgb.reshape(resolution * SCALING_FACTOR, resolution * SCALING_FACTOR, 3)

#     # init image with white background
#     rendered_image = torch.ones([resolution * SCALING_FACTOR, resolution * SCALING_FACTOR,3]).cuda()

#     mask_car = sdf < THRESHOLD
#     rendered_image[mask_car] = rgb[mask_car]

#     return rendered_image, mask_car



model_id = 0
image_id = 3



annotations = pickle.load(open(ANNOTATIONS_PATH, "rb"))
dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))
decoder_sdf = torch.load(DECODER_PATH).cuda()
decoder_rgb = torch.load(DECODER_PATH + "rgb").cuda()
decoder_sdf.eval()
decoder_rgb.train()

PLOT_PATH = "../../img_supervision/plots/decoder/intermediate_results/"

fig, axs = plt.subplots(4,4, figsize=(10,10))
for model_id in range(16):

    list_hash = list(dict_hash_2_code.keys())
    model_hash = list_hash[model_id]
    code_gt = dict_hash_2_code[model_hash].cuda()

    ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step = initialize_rendering(model_hash, image_id, annotations, IMAGES_PATH)
    ground_truth_image = np.array(ground_truth_image)

    ray_marching_vector = torch.tensor(ray_marching_vector,dtype=torch.float).cuda()
    pos_init_ray = torch.tensor(pos_init_ray,dtype=torch.float).cuda()
    min_step = torch.tensor(min_step,dtype=torch.float).cuda()
    max_step = torch.tensor(max_step,dtype=torch.float).cuda()


    # rendered_image, mask_car = ray_marching_rendering(decoder, code_gt, pos_init_ray, ray_marching_vector, min_step, max_step)
    rendered_image, mask_car = ray_marching_rendering(decoder_sdf, decoder_rgb, code_gt, pos_init_ray, ray_marching_vector, min_step, max_step)
    
    rendered_image = rendered_image.cpu().detach().numpy()
    min_step = min_step.cpu().detach().numpy()
    mask_car = mask_car.cpu().numpy()
    min_step = min_step.reshape(50,50)

    rendered_image[mask_car == False] = 0
    rendered_image[min_step == 0] = 1

    axs[model_id%4, (int)((model_id - model_id%4)/4)].imshow(rendered_image)

print("start saving")
fig.savefig(PLOT_PATH + "001.png")
print("done")
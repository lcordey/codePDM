""" 
Usefull functions
"""

import numpy as np
import torch
import pickle
import imageio
import cv2

from pytorch3d.ops.knn import knn_points
from skimage.transform import downscale_local_mean
from skimage import color

import IPython



RAY_MARCHING_RESOLUTION_PIXELS = 50
RAY_MARCHING_RESOLUTION_IMAGE = 250

BOUND_MAX_CUBE = 0.5
BOUND_MIN_CUBE = -0.5

THRESHOLD = 1/64
MAX_STEP = 1/16
MAX_ITER = 20
SCALING_FACTOR = 2


SCALING_RATIO_DU_TO_PADDING = 1/1.15625
OFFSET = torch.tensor([-0.05,0.005, -0.025]).cuda()

# SCALING_RATIO_DU_TO_PADDING = 1
# OFFSET = torch.tensor([0,0,0]).cuda()


def chamfer_distance_rgb(
    x,
    y,
    colors_x = None,
    colors_y = None
):

    """
    Chamfer distance between two pointclouds x and y.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! Only support batch of size 1 !!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
    """

    # checkt point inputs size
    if x.ndim == 3 and y.ndim == 3 and x.shape[0] == 1 and y.shape[0] == 1:
        pass
    elif x.ndim == 2 and y.ndim == 2:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    else:
        raise ValueError("Expected points to be of shape (1, P, D) or (P,D)")
    

    if colors_x is not None and colors_y is not None:
        return_cham_colors = True
    else:
        return_cham_colors = False
    
    # get dimension
    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")


    # checkt colors inputs size
    if return_cham_colors:
        if colors_x.ndim == 3 and colors_y.ndim == 3 and colors_x.shape[0] == 1 and colors_y.shape[0] == 1:
            colors_x = colors_x.squeeze()
            colors_y = colors_y.squeeze()
        elif colors_x.ndim == 2 and colors_y.ndim == 2:
            pass
        else:
            raise ValueError("Expected colors of points to be of shape (1, P, D) or (P,D)")

        if colors_x.shape[0] != P1 or colors_y.shape[0] != P2:
            raise ValueError("colors inputs size must match points size.")

        if colors_x.shape[1] != 3 or colors_y.shape[1] != 3:
            raise ValueError("last dimension or colors input should be of size 3 as it should be rgb values")

        if colors_x.max() > 1.0 or colors_x.min() < 0.0 or colors_y.max() > 1.0 or colors_y.min() < 0.0:
            raise ValueError("Colors values must be normalized between 0 and 1")
            


    # find nearest neighbours
    x_nn = knn_points(x, y, K=1)
    y_nn = knn_points(y, x, K=1)

    # get distance error from knn
    cham_x = x_nn.dists[..., 0]
    cham_y = y_nn.dists[..., 0]

    cham_x = cham_x.squeeze().sum()
    cham_y = cham_y.squeeze().sum()

    # average through all points
    cham_x = cham_x/P1
    cham_y = cham_y/P2

    cham_dist = cham_x + cham_y


    if return_cham_colors:

        # get lab colors too
        colors_x_lab = torch.tensor(color.rgb2lab(colors_x.cpu().numpy())).cuda()
        colors_y_lab = torch.tensor(color.rgb2lab(colors_y.cpu().numpy())).cuda()
        
        # find index from knn results
        idx_x = x_nn.idx[..., 0].squeeze()
        idx_y = y_nn.idx[..., 0].squeeze()

        # compute rgb error
        error_x_rgb = torch.nn.L1Loss()(colors_x[:], colors_y[idx_x])
        error_y_rgb = torch.nn.L1Loss()(colors_y[:], colors_x[idx_y])

        # normalize
        cham_colors_rgb = (error_x_rgb + error_y_rgb) / 2
        cham_colors_rgb = cham_colors_rgb * 255

        # compute lab error
        error_x_lab = torch.nn.L1Loss()(colors_x_lab[:], colors_y_lab[idx_x])
        error_y_lab = torch.nn.L1Loss()(colors_y_lab[:], colors_x_lab[idx_y])

        cham_colors_lab = (error_x_lab + error_y_lab) / 2
        
    else:
        cham_colors_rgb = None
        cham_colors_lab = None

    return cham_dist, cham_colors_rgb, cham_colors_lab



def convert_w2c(matrix_world_to_camera, frame, point):
    """ 
    convert a point in world coordinates into camera coordinates.
    """

    point_4d = np.resize(point, 4)
    point_4d[3] = 1
    co_local = matrix_world_to_camera.dot(point_4d)
    z = -co_local[2]

    f = np.empty([3,3])

    if z == 0.0:
        return np.array([0.5, 0.5, 0.0])
    else:
        for i in range(3):
            f[i] =  -(frame[i] / (frame[i][2]/z))

    min_x, max_x = f[2][0], f[1][0]
    min_y, max_y = f[1][1], f[0][1]

    x = (co_local[0] - min_x) / (max_x - min_x)
    y = (co_local[1] - min_y) / (max_y - min_y)

    return np.array([x,y,z])


def convert_view_to_camera_coordinates(frame, pixel_location):
    """ 
    convert a point in pixel coordinates into camera coordinates.
    """

    x = pixel_location[0]
    y = pixel_location[1]

    f = np.empty([3,3])
    camera_coordinate = np.empty([4])
    camera_coordinate[2] = -1 # z = 1
    camera_coordinate[3] = 1


    for i in range(3):
        f[i] =  -(frame[i] / (frame[i][2]))

    min_x, max_x = f[2][0], f[1][0]
    min_y, max_y = f[1][1], f[0][1]

    camera_coordinate[0] = x * (max_x - min_x) + min_x
    camera_coordinate[1] = y * (max_y - min_y) + min_y

    return camera_coordinate



def get_camera_matrix_and_frame(model_hash, image_id, annotations):
    """ 
    return fram and camera to object matrix, given a model, image, and the annotations file
    """
    
    frame = annotations[model_hash][image_id]['frame'].copy()
    matrix_object_to_world = annotations[model_hash][image_id]['matrix_object_to_world'].copy()
    matrix_world_to_camera = annotations["matrix_world_to_camera"]

    matrix_world_to_camera = matrix_world_to_camera[[1,0,2,3]]
    matrix_object_to_world = matrix_object_to_world[:,[1,0,2,3]][[2,1,0,3]]

    matrix_camera_to_world = np.linalg.inv(matrix_world_to_camera)
    matrix_world_to_object = np.linalg.inv(matrix_object_to_world)

    matrix_camera_to_object = matrix_world_to_object.dot(matrix_camera_to_world)

    return frame, matrix_camera_to_object


def initialize_rendering_image(model_hash, image_id, annotations, image_path):
    """ 
    return
    - The ground truth image to render
    - The initalized point to compute ray marching for each pixel.
    - The correpsonding vector that goes along the line or the ray marching process.
    - The minimum distance along which the algorithm will compute the ray marching process
    - The maximum distance along which the algorithm will compute the ray marching process
    """
    
    image_pth = image_path + model_hash + '/' + str(image_id) + '.png'
    ground_truth_image = np.array(imageio.imread(image_pth))/255

    resolution = RAY_MARCHING_RESOLUTION_IMAGE

    frame, matrix_camera_to_object = get_camera_matrix_and_frame(model_hash, image_id, annotations)

    cam_pos_cam_coord = np.array([0,0,0,1])
    cam_pos_obj_coord = matrix_camera_to_object.dot(cam_pos_cam_coord)

    pos_init_ray = np.ones([resolution * resolution, 3])
    pos_init_ray[:,0] *= cam_pos_obj_coord[0]
    pos_init_ray[:,1] *= cam_pos_obj_coord[1]
    pos_init_ray[:,2] *= cam_pos_obj_coord[2]

    ray_marching_vector = np.empty([resolution * resolution, 3])

    for i in range(resolution):
        for j in range(resolution):

            # select pixel to render
            pixel_pos = np.array([i/(resolution-1),j/(resolution-1)])

            # convert pixel in world coordinate
            pixel_pos_cam_coord = convert_view_to_camera_coordinates(frame, pixel_pos)
            pixel_pos_obj_coord = matrix_camera_to_object.dot(pixel_pos_cam_coord)

            ray_marching_vector[i * resolution + j,:] = ((pixel_pos_obj_coord - cam_pos_obj_coord)/np.linalg.norm(pixel_pos_obj_coord - cam_pos_obj_coord))[:3]


    min_step = np.zeros([resolution * resolution, 3])
    max_step = np.ones([resolution * resolution, 3]) * 1e38

    min_step[ray_marching_vector > 0] = (BOUND_MIN_CUBE - pos_init_ray[ray_marching_vector > 0]) / ray_marching_vector[ray_marching_vector > 0]
    min_step[ray_marching_vector < 0] = (BOUND_MAX_CUBE - pos_init_ray[ray_marching_vector < 0]) / ray_marching_vector[ray_marching_vector < 0]

    max_step[ray_marching_vector > 0] = (BOUND_MAX_CUBE - pos_init_ray[ray_marching_vector > 0]) / ray_marching_vector[ray_marching_vector > 0]
    max_step[ray_marching_vector < 0] = (BOUND_MIN_CUBE - pos_init_ray[ray_marching_vector < 0]) / ray_marching_vector[ray_marching_vector < 0]

    max_step[(ray_marching_vector == 0) * ~((pos_init_ray > BOUND_MIN_CUBE) * (pos_init_ray < BOUND_MAX_CUBE))] = 0

    min_step = min_step.max(1)
    max_step = max_step.min(1)

    max_step[min_step >= max_step] = 0
    min_step[min_step >= max_step] = 0

    return ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step


def initialize_rendering_pixels(model_hash, image_id, annotations, image_path, nb_samples):
    """ 
    return
    - The ground truth pixels to render (not all the image)
    - The initalized point to compute ray marching for each pixel.
    - The correpsonding vector that goes along the line or the ray marching process.
    - The minimum distance along which the algorithm will compute the ray marching process
    - The maximum distance along which the algorithm will compute the ray marching process
    """

    image_pth = image_path + model_hash + '/' + str(image_id) + '.png'
    ground_truth_image = np.array(imageio.imread(image_pth))/255
    ground_truth_pixels = np.empty([nb_samples, 3])

    height, width, _ = ground_truth_image.shape

    frame, matrix_camera_to_object = get_camera_matrix_and_frame(model_hash, image_id, annotations)

    cam_pos_cam_coord = np.array([0,0,0,1])
    cam_pos_obj_coord = matrix_camera_to_object.dot(cam_pos_cam_coord)

    pos_init_ray = np.ones([nb_samples, 3])
    pos_init_ray[:,0] *= cam_pos_obj_coord[0]
    pos_init_ray[:,1] *= cam_pos_obj_coord[1]
    pos_init_ray[:,2] *= cam_pos_obj_coord[2]

    ray_marching_vector = np.empty([nb_samples, 3])


    for i in range(nb_samples):

        sampling_res = 100

        # select pixel to render
        rnd_height = np.random.randint(height - 1)
        rnd_height_sampling = np.random.randint(sampling_res + 1)
        rnd_width = np.random.randint(width - 1)
        rnd_width_sampling = np.random.randint(sampling_res + 1)

        ground_truth_pixels[i] = ground_truth_image[rnd_height, rnd_width] * (sampling_res - rnd_height_sampling) * (sampling_res - rnd_width_sampling) \
                                + ground_truth_image[rnd_height + 1, rnd_width] * rnd_height_sampling * (sampling_res - rnd_width_sampling) \
                                + ground_truth_image[rnd_height, rnd_width + 1] * (sampling_res - rnd_height_sampling) * rnd_width_sampling \
                                + ground_truth_image[rnd_height + 1, rnd_width + 1] * rnd_height_sampling * rnd_width_sampling


        ground_truth_pixels[i] = ground_truth_pixels[i] / (sampling_res * sampling_res)

        pixel_pos_normalized = ((rnd_height * sampling_res + rnd_height_sampling) / ((height - 1) * sampling_res), (rnd_width * sampling_res + rnd_width_sampling) / ((width - 1) * sampling_res))



        # convert pixel in world coordinate
        pixel_pos_cam_coord = convert_view_to_camera_coordinates(frame, pixel_pos_normalized)
        pixel_pos_obj_coord = matrix_camera_to_object.dot(pixel_pos_cam_coord)

        ray_marching_vector[i,:] = ((pixel_pos_obj_coord - cam_pos_obj_coord)/np.linalg.norm(pixel_pos_obj_coord - cam_pos_obj_coord))[:3]


    min_step = np.zeros([nb_samples, 3])
    max_step = np.ones([nb_samples, 3]) * 1e38

    min_step[ray_marching_vector > 0] = (BOUND_MIN_CUBE - pos_init_ray[ray_marching_vector > 0]) / ray_marching_vector[ray_marching_vector > 0]
    min_step[ray_marching_vector < 0] = (BOUND_MAX_CUBE - pos_init_ray[ray_marching_vector < 0]) / ray_marching_vector[ray_marching_vector < 0]

    max_step[ray_marching_vector > 0] = (BOUND_MAX_CUBE - pos_init_ray[ray_marching_vector > 0]) / ray_marching_vector[ray_marching_vector > 0]
    max_step[ray_marching_vector < 0] = (BOUND_MIN_CUBE - pos_init_ray[ray_marching_vector < 0]) / ray_marching_vector[ray_marching_vector < 0]

    max_step[(ray_marching_vector == 0) * ~((pos_init_ray > BOUND_MIN_CUBE) * (pos_init_ray < BOUND_MAX_CUBE))] = 0

    min_step = min_step.max(1)
    max_step = max_step.min(1)

    max_step[min_step >= max_step] = 0
    min_step[min_step >= max_step] = 0

    return ground_truth_pixels, pos_init_ray, ray_marching_vector, min_step, max_step


def get_pos_from_ray_marching(decoder_sdf, latent_code, pos_init_ray, ray_marching_vector, min_step, max_step):
    """ 
    comute the ray marching process, and return the final position corresponding to each pixels
    """
    
    assert len(pos_init_ray) == len(ray_marching_vector) == len(min_step) == len(max_step), "Initialized parameters should have the same length"

    nb_samples = pos_init_ray.shape[0]

    # marching safely until the "cube" representing the zone where the sdf was trained, and therefore contain the whole object
    min_pos = pos_init_ray + min_step.unsqueeze(1).mul(ray_marching_vector)
    pos_along_ray = min_pos
    sum_step = min_step


    # only compute ray marching for ray passing through the cube
    mask_cube = (min_step != 0)
    mask_ray_still_marching = mask_cube

    # define ratio to accelerate ray marching
    ratio = torch.ones([nb_samples]).cuda()
    sdf = torch.zeros([nb_samples]).cuda()

    for iter in range(MAX_ITER):
        # compute sdf values

        if len(latent_code.shape) == 1:
            sdf[mask_ray_still_marching] = decoder_sdf(latent_code.unsqueeze(0).repeat([mask_ray_still_marching.count_nonzero(),1]), OFFSET + SCALING_RATIO_DU_TO_PADDING * pos_along_ray[mask_ray_still_marching]).squeeze().detach()
        else:
            sdf[mask_ray_still_marching] = decoder_sdf(latent_code[mask_ray_still_marching], OFFSET +  pos_along_ray[mask_ray_still_marching] * SCALING_RATIO_DU_TO_PADDING).squeeze().detach()

        # IPython.embed()

        # scaling: unit of 1 from the decoder's prediction correspond to a unit of 1 in the object coordinate
        sdf[mask_ray_still_marching] *= 1

        # compute the ratio using difference between old an new sdf values,
        if iter == 0:
            # store first sdf
            old_sdf = sdf
        else:
            # only compute ratio when the sdf is high enough and decreasing
            mask_ratio = (sdf > THRESHOLD) * (sdf < old_sdf)
            ratio[mask_ratio] = old_sdf[mask_ratio]/(old_sdf[mask_ratio] - sdf[mask_ratio])

            # store new sdf
            old_sdf[mask_ray_still_marching] = sdf[mask_ray_still_marching]
            
            # accelarating the step, this is only possible because the decoder tends to underestimate the distance due to the design of the loss
            sdf[mask_ratio] *= ratio[mask_ratio]

        # clamp value to prevent undesired issues
        sdf = sdf.clamp(None, MAX_STEP)

        # march along the ray
        step = sdf
        sum_step += step
        pos_along_ray = pos_along_ray + step.unsqueeze(1).mul(ray_marching_vector)

        mask_ray_still_marching = mask_ray_still_marching * (sum_step < max_step)


    return pos_along_ray


def render_pixels_from_pos(decoder_sdf, decoder_rgb, pos_along_ray, latent_code):
    """ 
    given the position computed from ray marching, return the colors of the pixels to render, as well as the silhouette of the object
    """
    
    # compute corresponding sdf
    sdf = decoder_sdf(latent_code, OFFSET + SCALING_RATIO_DU_TO_PADDING * pos_along_ray).squeeze().detach()
    rgb = decoder_rgb(latent_code, OFFSET + SCALING_RATIO_DU_TO_PADDING * pos_along_ray)

    rendered_pixels = torch.ones(rgb.shape).cuda()

    mask_car = sdf < THRESHOLD

    rendered_pixels[mask_car] = rgb[mask_car]

    return rendered_pixels, mask_car


def interpolate_final_pos(pos_along_ray, resolution=50, scaling_factor=2):
    """ 
    augment the resolution of the final image
    """
    
    # interpolate final position for a higher resolution
    pos_along_ray = pos_along_ray.reshape(resolution, resolution, 3).permute(2,0,1).unsqueeze(0)
    pos_along_ray = torch.nn.functional.interpolate(pos_along_ray, scale_factor = scaling_factor, mode='bilinear', align_corners=False)
    pos_along_ray = pos_along_ray.squeeze().permute(1,2,0).reshape(resolution * resolution * scaling_factor * scaling_factor, 3)

    return pos_along_ray

def render_image_from_pos(decoder_sdf, decoder_rgb, pos_along_ray, latent_code, resolution=50, scaling_factor=2):
    """ 
    given the position computed from ray marching, return the colors of the image to render, as well as the silhouette of the object
    """
    # compute corresponding sdf
    sdf = decoder_sdf(latent_code.unsqueeze(0).repeat([resolution * resolution * scaling_factor * scaling_factor,1]), OFFSET + SCALING_RATIO_DU_TO_PADDING * pos_along_ray).detach()
    rgb = decoder_rgb(latent_code.unsqueeze(0).repeat([resolution * resolution * scaling_factor * scaling_factor,1]), OFFSET + SCALING_RATIO_DU_TO_PADDING * pos_along_ray).detach()
    sdf = sdf.reshape(resolution * scaling_factor, resolution * scaling_factor)
    rgb = rgb.reshape(resolution * scaling_factor, resolution * scaling_factor, 3)

    # init image with white background
    rendered_image = torch.ones([resolution * scaling_factor, resolution * scaling_factor,3]).cuda()

    mask_car = sdf < THRESHOLD

    rendered_image[mask_car] = rgb[mask_car]


    return rendered_image, mask_car





# def initialize_rendering_from_3d_pos(model_hash, image_id, annotations, pos_3d):
#     nb_samples = len(pos_3d)

#     frame, matrix_camera_to_object = get_camera_matrix_and_frame(model_hash, image_id, annotations)

#     cam_pos_cam_coord = np.array([0,0,0,1])
#     cam_pos_obj_coord = matrix_camera_to_object.dot(cam_pos_cam_coord)

#     pos_init_ray = np.ones([nb_samples, 3])
#     pos_init_ray[:,0] *= cam_pos_obj_coord[0]
#     pos_init_ray[:,1] *= cam_pos_obj_coord[1]
#     pos_init_ray[:,2] *= cam_pos_obj_coord[2]

#     ray_marching_vector = np.empty([nb_samples, 3])

#     for i in range(nb_samples):
#         ray_marching_vector[i,:] = ((pos_3d[i] - cam_pos_obj_coord[:3])/np.linalg.norm(pos_3d[i] - cam_pos_obj_coord[:3]))

#     min_step = np.zeros([nb_samples, 3])
#     max_step = np.ones([nb_samples, 3]) * 1e38

#     min_step[ray_marching_vector > 0] = (BOUND_MIN_CUBE - pos_init_ray[ray_marching_vector > 0]) / ray_marching_vector[ray_marching_vector > 0]
#     min_step[ray_marching_vector < 0] = (BOUND_MAX_CUBE - pos_init_ray[ray_marching_vector < 0]) / ray_marching_vector[ray_marching_vector < 0]

#     max_step[ray_marching_vector > 0] = (BOUND_MAX_CUBE - pos_init_ray[ray_marching_vector > 0]) / ray_marching_vector[ray_marching_vector > 0]
#     max_step[ray_marching_vector < 0] = (BOUND_MIN_CUBE - pos_init_ray[ray_marching_vector < 0]) / ray_marching_vector[ray_marching_vector < 0]

#     max_step[(ray_marching_vector == 0) * ~((pos_init_ray > BOUND_MIN_CUBE) * (pos_init_ray < BOUND_MAX_CUBE))] = 0

#     min_step = min_step.max(1)
#     max_step = max_step.min(1)

#     max_step[min_step >= max_step] = 0
#     min_step[min_step >= max_step] = 0

#     return pos_init_ray, ray_marching_vector, min_step, max_step

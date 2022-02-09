import torch
import pickle
import imageio
import cv2
import numpy as np

from utils import *

import IPython

def convert_w2c(matrix_world_to_camera, frame, point):

    point_4d = np.resize(point, 4)
    point_4d[3] = 1
    co_local = matrix_world_to_camera.dot(point_4d)
    z = -co_local[2]

    if z == 0.0:
            return np.array([0.5, 0.5, 0.0])
    else:
        for i in range(3):
            frame[i] =  -(frame[i] / (frame[i][2]/z))

    min_x, max_x = frame[2][0], frame[1][0]
    min_y, max_y = frame[1][1], frame[0][1]

    x = (co_local[0] - min_x) / (max_x - min_x)
    y = (co_local[1] - min_y) / (max_y - min_y)

    return np.array([x,y,z])

class DatasetDecoderSDF(torch.utils.data.Dataset):
    "One epoch is num_model * num_samples_per_model"
    def __init__(self, list_hash, dict_gt_data, num_samples_per_model, dict_model_hash_2_idx):
        'Initialization'
        self.list_hash = list_hash
        self.dict_gt_data = dict_gt_data
        self.num_samples_per_model = num_samples_per_model
        self.dict_model_hash_2_idx = dict_model_hash_2_idx

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_hash) * self.num_samples_per_model

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        xyz_idx = index%self.num_samples_per_model
        model_hash = self.list_hash[(int)((index - xyz_idx)/self.num_samples_per_model)]
        # get model idx
        model_idx = self.dict_model_hash_2_idx[model_hash]

        # get sdf and rgb values
        sdf_gt = self.dict_gt_data["sdf"][model_hash][xyz_idx]
    ######################################## only used for testing ########################################
        rgb_gt = self.dict_gt_data["rgb"][model_hash][xyz_idx]
    ######################################## only used for testing ########################################

        return model_idx, sdf_gt, rgb_gt, xyz_idx

class DatasetDecoderTrainingRGB(torch.utils.data.Dataset):
    "One epoch is num_model * num_images_per_model"

    def __init__(self, list_hash, annotations, num_images_per_model, num_sample_per_image, dict_model_hash_2_idx, image_path):
        'Initialization'
        self.list_hash = list_hash
        self.annotations = annotations
        self.num_images_per_model = num_images_per_model
        self.num_sample_per_image = num_sample_per_image
        self.dict_model_hash_2_idx = dict_model_hash_2_idx
        self.image_path = image_path
        self.matrix_world_to_camera = annotations["matrix_world_to_camera"]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_hash) * self.num_images_per_model

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        image_id = index%self.num_images_per_model
        model_hash = self.list_hash[(int)((index - image_id)/self.num_images_per_model)]

        # get model idx
        model_idx = self.dict_model_hash_2_idx[model_hash]

        ground_truth_pixels, pos_init_ray, ray_marching_vector, min_step, max_step = initialize_rendering_pixels(model_hash, image_id, self.annotations, self.image_path, self.num_sample_per_image)

        return model_idx, ground_truth_pixels, pos_init_ray, ray_marching_vector, min_step, max_step



class DatasetDecoderValidationRGB(torch.utils.data.Dataset):
    "One epoch is num_model * num_images_per_model"

    def __init__(self, list_hash, annotations, num_images_per_model, dict_model_hash_2_idx, image_path):
        'Initialization'
        self.list_hash = list_hash
        self.annotations = annotations
        self.num_images_per_model = num_images_per_model
        self.dict_model_hash_2_idx = dict_model_hash_2_idx
        self.image_path = image_path
        self.matrix_world_to_camera = annotations["matrix_world_to_camera"]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_hash) * self.num_images_per_model

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        image_id = index%self.num_images_per_model
        model_hash = self.list_hash[(int)((index - image_id)/self.num_images_per_model)]

        # get model idx
        model_idx = self.dict_model_hash_2_idx[model_hash]

        ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step = initialize_rendering_image(model_hash, image_id, self.annotations, self.image_path)

        return model_idx, ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step


class DatasetLearnSDF(torch.utils.data.Dataset):
    "One epoch is num_model * num_image_per_model"
    def __init__(self, list_hash, annotations, num_images_per_model, image_path):
        'Initialization'
        self.list_hash = list_hash
        self.annotations = annotations
        self.num_images_per_model = num_images_per_model
        self.image_path = image_path

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_hash) * self.num_images_per_model


    def __getitem__(self, index):
        'Generates one sample of data'


        # Select sample
        image_id = index%self.num_images_per_model
        model_hash = self.list_hash[(int)((index - image_id)/self.num_images_per_model)]

        ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step = initialize_rendering_image(model_hash, image_id, self.annotations, self.image_path)


        return ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step


class DatasetGrid(torch.utils.data.Dataset):
    "One epoch is num_model * num_image_per_model"
    def __init__(self, list_hash, annotations, num_images_per_model, param_image, param_network, image_path):
        'Initialization'
        self.list_hash = list_hash
        self.annotations = annotations
        self.num_images_per_model = num_images_per_model
        self.width_image = param_image["width"]
        self.height_image = param_image["height"]
        self.num_slices = param_network["num_slices"]
        self.width_network = param_network["width"]
        self.height_network = param_network["height"]
        self.image_path = image_path
        self.matrix_world_to_camera = annotations["matrix_world_to_camera"]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_hash) * self.num_images_per_model

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        image_id = index%self.num_images_per_model
        model_hash = self.list_hash[(int)((index - image_id) / self.num_images_per_model)]

        # Load data and get label
        image_pth = self.image_path + model_hash + '/' + str(image_id) + '.png'
        input_im = imageio.imread(image_pth)

        # loc_2d = self.annotations[scene_id][rand_image_id]['2d'].copy()
        loc_3d = self.annotations[model_hash][image_id]['3d'].copy()
        frame = self.annotations[model_hash][image_id]['frame'].copy()

        # interpolate slices vertex coordinates
        loc_slice_3d = np.empty([self.num_slices,4,3])
        for i in range(self.num_slices):
            loc_slice_3d[i,0,:] = loc_3d[0,:] * (1-i/(self.num_slices-1)) + loc_3d[4,:] * i/(self.num_slices-1)
            loc_slice_3d[i,1,:] = loc_3d[1,:] * (1-i/(self.num_slices-1)) + loc_3d[5,:] * i/(self.num_slices-1)
            loc_slice_3d[i,2,:] = loc_3d[2,:] * (1-i/(self.num_slices-1)) + loc_3d[6,:] * i/(self.num_slices-1)
            loc_slice_3d[i,3,:] = loc_3d[3,:] * (1-i/(self.num_slices-1)) + loc_3d[7,:] * i/(self.num_slices-1)

        # convert to image plane coordinate
        loc_slice_2d = np.empty_like(loc_slice_3d)
        for i in range(self.num_slices):
            for j in range(4):
                    loc_slice_2d[i,j,:] = convert_w2c(self.matrix_world_to_camera, frame, loc_slice_3d[i,j,:]) 

        ###### y coordinate is inverted + rescaling #####
        loc_slice_2d[:,:,1] = 1 - loc_slice_2d[:,:,1]
        loc_slice_2d[:,:,0] = loc_slice_2d[:,:,0] * self.width_image
        loc_slice_2d[:,:,1] = loc_slice_2d[:,:,1] * self.height_image

        # grid to give as input to the network
        input_grid = np.empty([self.num_slices, self.width_network, self.height_network, 3])

        # fill grid by slices
        for i in range(self.num_slices):
            src = loc_slice_2d[i,:,:2].copy()
            dst = np.array([[0, self.height_network], [self.width_network, self.height_network], [self.width_network, 0], [0,0]])
            h, mask = cv2.findHomography(src, dst)
            slice = cv2.warpPerspective(input_im, h, (self.width_network,self.height_network))
            input_grid[i,:,:,:] = slice

        # rearange, normalize and convert to tensor
        input_grid = np.transpose(input_grid, [3,0,1,2])
        input_grid = input_grid/255 - 0.5
        input_grid = torch.tensor(input_grid, dtype = torch.float)

        return input_grid, model_hash



class DatasetDecoderSDF2(torch.utils.data.Dataset):
    "One epoch is num_model * num_samples_per_model"
    def __init__(self, data):
        'Initialization'
        self.data = data
        self.num_samples = len(data)
    def __len__(self):
        'Denotes the total number of samples'
        return self.num_samples

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        xyz_idx = index%self.num_samples

        # get sdf and rgb values
        sdf_gt = self.data[xyz_idx]

        return sdf_gt, xyz_idx


class DatasetDecoderSDF3(torch.utils.data.Dataset):
    "One epoch is num_model * num_samples_per_model"
    def __init__(self, num_samples, num_images, matrix_world_to_camera, frame, model_annotations):
        'Initialization'
        self.num_samples = num_samples
        self.num_images = num_images
        self.matrix_world_to_camera = matrix_world_to_camera
        self.frame = frame
        self.model_annotations = model_annotations

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_samples

    def __getitem__(self, index):
        'Generates one sample of data'

        sdf_estimation = -1e42

        pos_3d_ojbect = np.random.rand(3) - 0.5

        for image_id in range(self.num_images):

            matrix_object_to_world = self.model_annotations[image_id]['matrix_object_to_world'].copy()
            matrix_object_to_world = matrix_object_to_world[:,[1,0,2,3]][[2,1,0,3]]

            pos_3d_world = matrix_object_to_world[:3,:3].dot(pos_3d_ojbect)
            pos_2d = convert_w2c(self.matrix_world_to_camera, self.frame, pos_3d_world)
            pos_2d[:2] *= 300

            try:
                img_dist = self.model_annotations[image_id]["2d_sdf"][pos_2d[0].astype(int), pos_2d[1].astype(int)]
            except:
                img_dist = 1e-10

            result_from_image = img_dist * pos_2d[2]

            sdf_estimation = max(sdf_estimation, result_from_image)

        return pos_3d_world, sdf_estimation



class DatasetDecoderTrainingRGB_2(torch.utils.data.Dataset):
    "One epoch is num_model * num_images_per_model"

    def __init__(self, model_hash, num_repetition_image_per_epoch, num_images_per_model, num_sample_per_image, annotations, image_path):
        'Initialization'
        self.model_hash = model_hash
        self.num_repetition_image_per_epoch = num_repetition_image_per_epoch
        self.num_images_per_model = num_images_per_model
        self.num_sample_per_image = num_sample_per_image
        self.annotations = annotations
        self.matrix_world_to_camera = annotations["matrix_world_to_camera"]
        self.image_path = image_path

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_repetition_image_per_epoch * self.num_images_per_model

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image_id = index%self.num_images_per_model
        ground_truth_pixels, pos_init_ray, ray_marching_vector, min_step, max_step = initialize_rendering_pixels(self.model_hash, image_id, self.annotations, self.image_path, self.num_sample_per_image)

        return ground_truth_pixels, pos_init_ray, ray_marching_vector, min_step, max_step



class DatasetDecoderValidationRGB_2(torch.utils.data.Dataset):
    "One epoch is num_model * num_images_per_model"

    def __init__(self, model_hash, num_images_validation, annotations, image_path):
        'Initialization'
        self.model_hash = model_hash
        self.num_images_validation = num_images_validation
        self.annotations = annotations
        self.matrix_world_to_camera = annotations["matrix_world_to_camera"]
        self.image_path = image_path

    def __len__(self):
        'Denotes the total number of samples'
        return self.num_images_validation

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        image_id = index
        ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step = initialize_rendering_image(self.model_hash, image_id, self.annotations, self.image_path)

        return ground_truth_image, pos_init_ray, ray_marching_vector, min_step, max_step

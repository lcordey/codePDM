import torch
import imageio
import random
import pickle
import numpy as np
import cv2
import time

IMAGES_PATH = "../../image2sdf/input_images/images/"
MATRIX_PATH = "../../image2sdf/input_images/matrix_w2c.pkl"


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

class DatasetGrid(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dict_labels, target_code, annotations, start_validation_id, end_validation_id, width_input, height_input, num_slices, width_input_network, height_input_network):
        'Initialization'
        self.dict = dict_labels
        self.code = target_code
        self.annotations = annotations
        self.start_id = start_validation_id
        self.end_id = end_validation_id
        self.width_input = width_input
        self.height_input = height_input
        self.num_slices = num_slices
        self.width_input_network = width_input_network
        self.height_input_network = height_input_network
        self.list_IDs = list_IDs
        self.matrix_world_to_camera = pickle.load(open(MATRIX_PATH, 'rb'))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        scene_id = self.list_IDs[index]

        # Load data and get label
        rand_image_id = random.randint(self.start_id, self.end_id - 1)
        image_pth = IMAGES_PATH + scene_id + '/' + str(rand_image_id) + '.png'
        input_im = imageio.imread(image_pth)

        # loc_2d = self.annotations[scene_id][rand_image_id]['2d'].copy()
        loc_3d = self.annotations[scene_id][rand_image_id]['3d'].copy()
        frame = self.annotations[scene_id][rand_image_id]['frame'].copy()

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
        loc_slice_2d[:,:,0] = loc_slice_2d[:,:,0] * self.width_input
        loc_slice_2d[:,:,1] = loc_slice_2d[:,:,1] * self.height_input

        # grid to give as input to the network
        input_grid = np.empty([self.num_slices, self.width_input_network, self.height_input_network, 3])


        # fill grid by slices
        for i in range(self.num_slices):
            src = loc_slice_2d[i,:,:2].copy()
            dst = np.array([[0, self.height_input_network], [self.width_input_network, self.height_input_network], [self.width_input_network, 0], [0,0]])
            h, mask = cv2.findHomography(src, dst)
            slice = cv2.warpPerspective(input_im, h, (self.width_input_network,self.height_input_network))
            input_grid[i,:,:,:] = slice

        # rearange, normalize and convert to tensor
        input_grid = np.transpose(input_grid, [3,0,1,2])
        input_grid = input_grid/255 - 0.5
        input_grid = torch.tensor(input_grid, dtype = torch.float)

        # target code
        target_code = self.code[self.dict[scene_id]]

        return input_grid, target_code


class DatasetFace(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_IDs, dict_labels, target_code, annotations,
                 start_validation_id, end_validation_id,
                 width_input, height_input,
                 width_input_network, height_input_network, depth_input_network):
        'Initialization'
        self.dict = dict_labels
        self.code = target_code
        self.annotations = annotations
        self.start_id = start_validation_id
        self.end_id = end_validation_id
        self.width_input = width_input
        self.height_input = height_input
        self.width_input_network = width_input_network
        self.height_input_network = height_input_network
        self.depth_input_network = depth_input_network
        self.list_IDs = list_IDs
        self.matrix_world_to_camera = pickle.load(open(MATRIX_PATH, 'rb'))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        scene_id = self.list_IDs[index]

        # Load data and get label
        rand_image_id = random.randint(self.start_id, self.end_id - 1)
        image_pth = IMAGES_PATH + scene_id + '/' + str(rand_image_id) + '.png'
        input_im = imageio.imread(image_pth)

        loc_2d = self.annotations[scene_id][rand_image_id]['2d'].copy()
        # loc_3d = self.annotations[scene_id][rand_image_id]['3d'].copy()
        # frame = self.annotations[scene_id][rand_image_id]['frame'].copy()

        ###### y coordinate is inverted + rescaling #####
        loc_2d[:,1] = 1 - loc_2d[:,1]
        loc_2d[:,0] = loc_2d[:,0] * self.width_input
        loc_2d[:,1] = loc_2d[:,1] * self.height_input


        # front
        src = np.array([loc_2d[0,:2],loc_2d[1,:2],loc_2d[2,:2],loc_2d[3,:2]]).copy()
        dst = np.array([[0,self.height_input_network],[self.width_input_network,self.height_input_network],[self.width_input_network,0],[0,0]])
        h, mask = cv2.findHomography(src, dst)
        front = cv2.warpPerspective(input_im, h, (self.width_input_network,self.height_input_network))

        # left
        src = np.array([loc_2d[1,:2],loc_2d[5,:2],loc_2d[6,:2],loc_2d[2,:2]]).copy()
        dst = np.array([[0,self.height_input_network],[self.depth_input_network,self.height_input_network],[self.depth_input_network,0],[0,0]])
        h, mask = cv2.findHomography(src, dst)
        left = cv2.warpPerspective(input_im, h, (self.depth_input_network,self.height_input_network))

        # back
        src = np.array([loc_2d[5,:2],loc_2d[4,:2],loc_2d[7,:2],loc_2d[6,:2]]).copy()
        dst = np.array([[0,self.height_input_network],[self.width_input_network,self.height_input_network],[self.width_input_network,0],[0,0]])
        h, mask = cv2.findHomography(src, dst)
        back = cv2.warpPerspective(input_im, h, (self.width_input_network,self.height_input_network))

        # right
        src = np.array([loc_2d[4,:2],loc_2d[0,:2],loc_2d[3,:2],loc_2d[7,:2]]).copy()
        dst = np.array([[0,self.height_input_network],[self.depth_input_network,self.height_input_network],[self.depth_input_network,0],[0,0]])
        h, mask = cv2.findHomography(src, dst)
        right = cv2.warpPerspective(input_im, h, (self.depth_input_network,self.height_input_network))

        # top
        src = np.array([loc_2d[3,:2],loc_2d[2,:2],loc_2d[6,:2],loc_2d[7,:2]]).copy()
        dst = np.array([[0,self.depth_input_network],[self.width_input_network,self.depth_input_network],[self.width_input_network,0],[0,0]])
        h, mask = cv2.findHomography(src, dst)
        top = cv2.warpPerspective(input_im, h, (self.width_input_network,self.depth_input_network))

        

        # rearange, normalize and convert to tensor
        front = np.transpose(front, [3,0,1,2])
        front = front/255 - 0.5
        front = torch.tensor(front, dtype = torch.float)

        left = np.transpose(left, [3,0,1,2])
        left = left/255 - 0.5
        left = torch.tensor(left, dtype = torch.float)

        back = np.transpose(back, [3,0,1,2])
        back = back/255 - 0.5
        back = torch.tensor(back, dtype = torch.float)

        right = np.transpose(right, [3,0,1,2])
        right = right/255 - 0.5
        right = torch.tensor(right, dtype = torch.float)

        top = np.transpose(top, [3,0,1,2])
        top = top/255 - 0.5
        top = torch.tensor(top, dtype = torch.float)

        # target code
        target_code = self.code[self.dict[scene_id]]

        return front, left, back, right, top, target_code
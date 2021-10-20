import torch
import imageio
import random
import numpy as np

IMAGES_PATH = "../../image2sdf/input_images/images/"

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, dict_labels, target_code, annotations, start_validation_id, end_validation_id, width_image, height_image):
        'Initialization'
        self.dict = dict_labels
        self.code = target_code
        self.annotations = annotations
        self.start_id = start_validation_id
        self.end_id = end_validation_id
        self.width_image = width_image
        self.height_image = height_image
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        rand_image_id = random.randint(self.start_id, self.end_id - 1)
        image_pth = IMAGES_PATH + ID + '/' + str(rand_image_id) + '.png'
        input_im = imageio.imread(image_pth)
        input_im = np.transpose(input_im,(2,0,1))


        input_loc = np.empty([20])

        for loc, loc_id in zip(self.annotations[ID][rand_image_id].keys(), range(len(self.annotations[ID][rand_image_id].keys()))):
            if loc[-1] == 'x' or loc[-5:] == 'width':
                input_loc[loc_id] = self.annotations[ID][rand_image_id][loc]/self.width_image
            else:
                input_loc[loc_id] = self.annotations[ID][rand_image_id][loc]/self.height_image

        input_im = input_im/255 - 0.5
        input_loc = input_loc - 0.5

        input_im = torch.tensor(input_im, dtype = torch.float)
        input_loc = torch.tensor(input_loc, dtype = torch.float)

        target_code = self.code[self.dict[ID]]


        return input_im, input_loc, target_code
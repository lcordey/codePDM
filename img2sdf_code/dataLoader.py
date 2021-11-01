from numpy.core.fromnumeric import _std_dispatcher
import torch
import numpy as np
import h5py

class DatasetDecoder(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_hash, sdf_dir_pth, resolution, num_samples_per_model):
        'Initialization'
        self.list_hash = list_hash
        self.sdf_dir_pth = sdf_dir_pth
        self.resolution = resolution
        self.num_samples_per_model = num_samples_per_model

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_hash)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        model_hash = self.list_hash[index]

        h5f = h5py.File(self.sdf_dir_pth + model_hash + '.h5', 'r')
        h5f_tensor = torch.tensor(h5f["tensor"][()], dtype = torch.float)

        num_total_point_in_model = (int)(h5f_tensor.numel()/4)

        sdf_gt = np.reshape(h5f_tensor[:,:,:,0], [num_total_point_in_model])
        rgb_gt = np.reshape(h5f_tensor[:,:,:,1:], [num_total_point_in_model , 3])

        xyz_idx = np.random.randint(num_total_point_in_model, size = self.num_samples_per_model)

        sdf_gt = sdf_gt[xyz_idx]
        rgb_gt = rgb_gt[xyz_idx]

        sdf_gt = sdf_gt / self.resolution
        rgb_gt = rgb_gt / 255


        return model_hash, sdf_gt, rgb_gt, xyz_idx
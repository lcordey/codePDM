import torch
import numpy as np
import time


class DatasetDecoder(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, list_hash, dict_gt_data, num_samples_per_model, dict_model_hash_2_idx):
        'Initialization'
        self.list_hash = list_hash
        self.dict_gt_data = dict_gt_data
        self.num_samples_per_model = num_samples_per_model
        self.dict_model_hash_2_idx = dict_model_hash_2_idx

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_hash) * self.num_samples_per_model
        # return len(self.list_hash)

    def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        xyz_idx = index%self.num_samples_per_model
        model_hash = self.list_hash[(int)((index - xyz_idx)/self.num_samples_per_model)]
        model_idx = self.dict_model_hash_2_idx[model_hash]

        sdf_gt = self.dict_gt_data["sdf"][model_hash][xyz_idx]
        rgb_gt = self.dict_gt_data["rgb"][model_hash][xyz_idx]


        # model_hash = self.list_hash[index]
        # model_idx = self.dict_model_hash_2_idx[model_hash]

        # get model gt data from dictionnary
        # sdf_gt = self.dict_gt_data["sdf"][model_hash]
        # rgb_gt = self.dict_gt_data["rgb"][model_hash]
        
        # num_total_point_in_model = sdf_gt.shape[0]

        # Select random sample from model
        # xyz_idx = np.random.randint(num_total_point_in_model, size = self.num_samples_per_model)
        # sdf_gt = sdf_gt[xyz_idx]
        # rgb_gt = rgb_gt[xyz_idx]

        return model_idx, sdf_gt, rgb_gt, xyz_idx

from torch.utils.data import Dataset
import os
import scipy.io as sio
import numpy as np

class SeisDataset(Dataset):
    def __init__(
        self,
        gt_path,
        mask_path,
        test_mask_path,
    ):
        super().__init__()
        self.gt_path = gt_path
        self.mask_path = mask_path
        self.test_mask_path = test_mask_path

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        gt_path = self.gt_path
        data = sio.loadmat(gt_path)
        data = data['data']
        gt_data = data[np.newaxis, ...]
        gt_data = gt_data.astype(np.float32)

        mask_path = self.mask_path
        mask_data = sio.loadmat(mask_path)
        mask_data = mask_data['data']
        mask_data = mask_data[np.newaxis, ...]
        miss_position = mask_data.astype(np.float32)
        miss_data = np.multiply(miss_position, gt_data)

        test_position = miss_position
        test_data = miss_data

        return miss_position, miss_data, test_position, test_data, gt_data
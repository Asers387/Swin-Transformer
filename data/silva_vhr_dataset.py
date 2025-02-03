import h5py
import json
from torch.nn import functional as F

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


class SilvaVHR(Dataset):
    def __init__(self, root_path, split_path, split_name, transform, target_transform=None):
        super().__init__()
        
        # Potential fix for efficient multiprocessing with h5py
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        self.data = None
        self.data_path = Path(root_path) / split_path / 'vhr-silva.hdf5'
        self.split_name = split_name

        data = h5py.File(self.data_path, 'r')[self.split_name]
        self.len_data = len(data)

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.len_data
                
    def __getitem__(self, idx):
        if self.data is None:
            self.data = h5py.File(self.data_path, 'r')[self.split_name]

        image_vhr, mask_lr = self.transform(self.data, idx)
        image_lr = F.interpolate(image_vhr.unsqueeze(dim=0), scale_factor=(0.125, 0.125), mode='area').squeeze(dim=0)
        return image_lr, image_vhr, mask_lr
    

def get_mean_std(root_path, split_path):
    with open(Path(root_path) / split_path / 'metadata.json', 'r') as f_r:
        metadata = json.loads(f_r.read())

    mean = metadata['train_mean']
    std = metadata['train_std']

    return mean, std

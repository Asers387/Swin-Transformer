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
        
        self.root_path = Path(root_path)

        # TODO check if dataloader uses processes, as threads don't work
        self._f_data = h5py.File(self.root_path / split_path / 'vhr-silva.hdf5', 'r')
        # self._f_data = h5py.File(self.root_path / split_path / 'vhr-silva.hdf5', 'r', driver='mpio')
        # self._f_data = h5py.File(self.root_path / split_path / 'vhr-silva.hdf5', 'r', swmr=True)
        self.data = self._f_data[split_name]

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        image_vhr, mask_lr = self.transform(self.data, idx)
        image_lr = F.interpolate(image_vhr.unsqueeze(dim=0), scale_factor=(0.125, 0.125), mode='area').squeeze(dim=0)
        return image_lr, image_vhr, mask_lr
    

def get_mean_std(root_path, split_path):
    with open(Path(root_path) / split_path / 'metadata.json', 'r') as f_r:
        metadata = json.loads(f_r.read())

    mean = metadata['train_mean']
    std = metadata['train_std']

    return mean, std

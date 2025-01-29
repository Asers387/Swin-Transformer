import h5py
import json

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


class SilvaVHR(Dataset):
    def __init__(self, root_path, split_path, split_name, transform, target_transform=None):
        super().__init__()
        
        self.root_path = Path(root_path)

        self._f_data = h5py.File(self.root_path / split_path / 'vhr-silva.hdf5', 'r')
        self.data = self._f_data[split_name]

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        image, mask = self.transform(self.data, idx)
        return image, mask
    

def get_mean_std(root_path, split_path):
    with open(Path(root_path) / split_path / 'metadata.json', 'r') as f_r:
        metadata = json.loads(f_r.read())

    mean = metadata['train_mean']
    std = metadata['train_std']

    return mean, std

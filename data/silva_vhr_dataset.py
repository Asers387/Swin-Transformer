import json

from pathlib import Path
from PIL import Image
from tifffile import TiffFile
from torch.nn import functional as F
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


class SilvaVHR(Dataset):
    def __init__(self, root_path, split_path, split_name, transform, target_transform=None):
        super().__init__()
        
        self.root_path = Path(root_path)

        tiff_path = self.root_path / split_path / 'tiff'

        split_json_name = self.root_path / split_path / f'{split_name}.json'
        with open(split_json_name, 'r') as f:
            data = json.loads(f.read())
            
            self.data = []
            for datum in data:
                datum_path = Path(datum['image_vhr'])
                self.data.append(str(tiff_path / datum_path.parent.name / f'{datum_path.stem}.tif'))

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data)
                
    def __getitem__(self, idx):
        with TiffFile(self.data[idx]) as f:
            image_vhr, mask_lr = self.transform(f.pages[0])

        image_lr = F.interpolate(image_vhr.unsqueeze(dim=0), scale_factor=(0.125, 0.125), mode='area').squeeze(dim=0)
        return image_lr, image_vhr, mask_lr


def get_mean_std(root_path, split_path):
    with open(Path(root_path) / split_path / 'metadata.json', 'r') as f_r:
        metadata = json.loads(f_r.read())

    mean = metadata['train_mean']
    std = metadata['train_std']

    return mean, std

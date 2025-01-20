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

        split_json_name = (self.root_path / split_path / split_name).with_suffix('.json')
        with open(split_json_name, 'r') as f:
            self.data = json.loads(f.read())

        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.data)
    
    def get_image(self, image_path):
        image = Image.open(image_path)
        image, mask = self.transform(image)
        return image, mask
            
    def __getitem__(self, idx):
        datum = self.data[idx]
        image, mask = self.get_image(self.root_path / datum['image_lr'])
        return image, mask
    

def get_mean_std(root_path, split_path):
    with open(Path(root_path) / split_path / 'metadata.json', 'r') as f_r:
        metadata = json.loads(f_r.read())

    mean = metadata['train_mean']
    std = metadata['train_std']

    return mean, std

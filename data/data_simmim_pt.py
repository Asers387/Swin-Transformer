# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import numpy as np

import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate

from data.silva_vhr_dataset import SilvaVHR, get_mean_std


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6, random_mask=True):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

        self.random_mask = random_mask
        if not self.random_mask:
            self.static_mask = np.random.permutation(self.token_count)[:self.mask_count]
        
    def __call__(self):
        if self.random_mask:
            mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        else:
            mask_idx = self.static_mask

        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


class SimMIMTransform:
    def __init__(self, split_name, config):
        if config.MODEL.TYPE in ['swin', 'swinv2']:
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        else:
            raise NotImplementedError
        
        mean, std = get_mean_std(config.DATA.DATA_PATH, config.DATA.SPLIT_PATH)

        if split_name == 'train':
            self.transform_img = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
            random_mask = True
        elif split_name == 'val' or split_name == 'test':
            self.transform_img = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.CenterCrop(config.DATA.IMG_SIZE),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])
            random_mask = False
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
            random_mask=random_mask
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return img, mask


def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret


def build_loader_simmim(split_name, config):
    transform = SimMIMTransform(split_name, config)
    dataset = SilvaVHR(config.DATA.DATA_PATH, config.DATA.SPLIT_PATH, split_name, transform)

    if split_name == 'train':
        shuffle = True
        drop_last = True
    elif split_name == 'val' or split_name == 'test':
        shuffle = False
        drop_last = False
    else:
        raise NotImplementedError
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle
    )

    dataloader = DataLoader(
        dataset,
        config.DATA.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    return dataloader


def build_loaders_simmim(config):
    dataloader_train = build_loader_simmim('train', config)
    dataloader_val = build_loader_simmim('val', config)
    dataloader_test = build_loader_simmim('test', config)
    
    return dataloader_train, dataloader_val, dataloader_test

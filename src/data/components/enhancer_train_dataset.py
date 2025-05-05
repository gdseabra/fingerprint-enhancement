import os

import torch
import wsq
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class EnhancerTrainDataset(Dataset):
    def __init__(self, data_dir: str = "data/", data_list: str = None, transform=None, gabor_transform = None, lat_subdir = '/latents/', ref_subdir = '/references/', gabor_subdir = '/gabor/', mask_subdir = '/masks/', apply_mask = 0):
        self.data_dir        = data_dir
        self.transform       = transform
        self.gabor_transform = gabor_transform
        self.data_list       = data_dir + data_list if data_list is not None else data_dir + "/data_list.txt"

        with open(self.data_list) as fp:
            lines = fp.readlines()

        self.data = [line.strip() for line in lines]

        self.lat_suffix   = "." + os.listdir(data_dir + lat_subdir)[0].split(".")[-1]
        self.ref_suffix   = "." + os.listdir(data_dir + ref_subdir)[0].split(".")[-1]
        self.gabor_suffix = "." + os.listdir(data_dir + gabor_subdir)[0].split(".")[-1]
        self.mask_suffix  = "." + os.listdir(data_dir + mask_subdir)[0].split(".")[-1]


        self.lat_subdir   = lat_subdir
        self.ref_subdir   = ref_subdir
        self.gabor_subdir = gabor_subdir
        self.mask_subdir  = mask_subdir

        self.apply_mask = apply_mask



    def __getitem__(self, ix):
        lat   = Image.open(self.data_dir + self.lat_subdir   + self.data[ix] + self.lat_suffix)
        mask  = Image.open(self.data_dir + self.mask_subdir  + self.data[ix] + self.mask_suffix)

        try:
            ref   = Image.open(self.data_dir + self.ref_subdir   + self.data[ix] + self.ref_suffix)
            gabor = Image.open(self.data_dir + self.gabor_subdir + self.data[ix] + self.gabor_suffix)


        except FileNotFoundError: # especial case when are dealing with an synthetic augmented dataset
            ref   = Image.open(self.data_dir + self.ref_subdir   + self.data[ix].split('_')[0] + self.ref_suffix)
            gabor = Image.open(self.data_dir + self.gabor_subdir + self.data[ix].split('_')[0] + self.gabor_suffix)


        # normalizing lat and ref to -1, 1

        lat = transforms.ToTensor()(lat)
        ref = transforms.ToTensor()(ref)


        lat_mean = torch.mean(lat)
        lat_std  = torch.std(lat)

        lat = transforms.Normalize(mean=[lat_mean], std=[2 * lat_std])(lat)



        ref_mean = torch.mean(ref)
        ref_std  = torch.std(ref)


        ref = transforms.Normalize(mean=[ref_mean], std=[2 * ref_std])(ref)

        if self.gabor_transform:
            gabor = self.gabor_transform(gabor)
            mask  = self.gabor_transform(mask)
        
        ref_white = torch.max(ref)
        gab_white = torch.max(gabor)

        
        if self.apply_mask:
            ref   = torch.where(mask == 0, ref_white, ref)
            gabor = torch.where(mask == 0, gab_white, gabor)

        return lat, torch.concat([ref, gabor, mask], axis=0)

    def __len__(self):
        return len(self.data)

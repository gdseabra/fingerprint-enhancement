import os

import torch
import wsq
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class EnhancerTrainDataset(Dataset):
    def __init__(self, data_dir: str = "data/", data_list: str = None, transform=None, skel_transform = None, lat_subdir = '/latents/', ref_subdir = '/references/', skel_subdir = '/skel/', mask_subdir = '/masks/', mnt_map_subdir='mnt_map', apply_mask = 0):
        self.data_dir        = data_dir
        self.transform       = transform
        self.skel_transform = skel_transform
        self.data_list       = data_dir + data_list if data_list is not None else data_dir + "/data_list.txt"

        with open(self.data_list) as fp:
            lines = fp.readlines()

        self.data = [line.strip() for line in lines]

        self.lat_suffix   = "." + os.listdir(data_dir + lat_subdir)[0].split(".")[-1]
        self.ref_suffix   = "." + os.listdir(data_dir + ref_subdir)[0].split(".")[-1]
        self.skel_suffix = "." + os.listdir(data_dir + skel_subdir)[0].split(".")[-1]
        self.mnt_map_suffix = "." + os.listdir(data_dir + mnt_map_subdir)[0].split(".")[-1]
        self.mask_suffix  = "." + os.listdir(data_dir + mask_subdir)[0].split(".")[-1]


        self.lat_subdir   = lat_subdir
        self.ref_subdir   = ref_subdir
        self.skel_subdir = skel_subdir
        self.mask_subdir  = mask_subdir
        self.mnt_map_subdir = mnt_map_subdir

        self.apply_mask = apply_mask



    def __getitem__(self, ix):
        lat   = Image.open(self.data_dir+self.lat_subdir+self.data[ix]+self.lat_suffix)
        mask  = Image.open(self.data_dir + self.mask_subdir  + self.data[ix] + self.mask_suffix)

        try:
            ref   = Image.open(self.data_dir + self.ref_subdir   + self.data[ix] + self.ref_suffix)
            skel = Image.open(self.data_dir + self.skel_subdir + self.data[ix] + self.skel_suffix)
            mnt_map = Image.open(self.data_dir + self.mnt_map_subdir + self.data[ix] + self.mnt_map_suffix)


        except FileNotFoundError: # especial case when are dealing with an synthetic augmented dataset
            ref   = Image.open(self.data_dir + self.ref_subdir   + self.data[ix].split('_')[0] + self.ref_suffix)
            skel = Image.open(self.data_dir + self.skel_subdir + self.data[ix].split('_')[0] + self.skel_suffix)
            mnt_map = Image.open(self.data_dir + self.mnt_map_subdir + self.data[ix].split('_')[0] + self.mnt_map_suffix)


        # normalizing lat and ref to -1, 1

        lat = transforms.ToTensor()(lat)
        ref = transforms.ToTensor()(ref)


        lat_mean = torch.mean(lat)
        lat_std  = torch.std(lat)

        lat = transforms.Normalize(mean=[lat_mean], std=[2 * lat_std])(lat)


        ref_mean = torch.mean(ref)
        ref_std  = torch.std(ref)


        ref = transforms.Normalize(mean=[ref_mean], std=[2 * ref_std])(ref)

        if self.skel_transform:
            skel = self.skel_transform(skel)
            mask  = self.skel_transform(mask)
            mnt_map = self.skel_transform(mnt_map)
        
        ref_white = ref.max()
        gab_white = 0

        
        ref   = torch.where(mask == 0, ref_white, ref)
        skel = torch.where(mask == 0, gab_white, skel)

        return lat, torch.concat([ref, skel, mask, mnt_map], axis=0)

    def __len__(self):
        return len(self.data)

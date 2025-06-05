import os

import torch
import wsq
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import numpy as np

class EnhancerTrainDataset(Dataset):
    def __init__(self, 
        data_dir: str = "data/", 
        data_list: str = None, 
        transform=None, 
        skel_transform=None, 
        patch_size=None, 
        lat_subdir = '/latents/', 
        ref_subdir = '/references/', 
        skel_subdir = '/skel/', 
        bin_subdir = '/bin/', 
        mask_subdir = '/masks/', 
        mnt_map_subdir='mnt_map', 
        apply_mask = 0
    ):
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
        self.bin_suffix = "." + os.listdir(data_dir + bin_subdir)[0].split(".")[-1]
        self.mnt_map_suffix = "." + os.listdir(data_dir + mnt_map_subdir)[0].split(".")[-1]
        self.mask_suffix  = "." + os.listdir(data_dir + mask_subdir)[0].split(".")[-1]
        
        self.lat_subdir   = lat_subdir
        self.ref_subdir   = ref_subdir
        self.skel_subdir = skel_subdir
        self.bin_subdir = bin_subdir
        self.mask_subdir  = mask_subdir
        self.mnt_map_subdir = mnt_map_subdir

        self.apply_mask = apply_mask

        self.patch_size = patch_size



    def __getitem__(self, ix):
        lat   = Image.open(self.data_dir+self.lat_subdir+self.data[ix]+self.lat_suffix)
        mask  = Image.open(self.data_dir + self.mask_subdir  + self.data[ix] + self.mask_suffix)

        try:
            ref   = Image.open(self.data_dir + self.ref_subdir   + self.data[ix] + self.ref_suffix)
            bin = Image.open(self.data_dir + self.bin_subdir + self.data[ix] + self.bin_suffix)
            # skel = Image.open(self.data_dir + self.skel_subdir + self.data[ix] + self.skel_suffix)
            # mnt_map = Image.open(self.data_dir + self.mnt_map_subdir + self.data[ix] + self.mnt_map_suffix)


        except FileNotFoundError: # especial case when are dealing with an synthetic augmented dataset
            ref   = Image.open(self.data_dir + self.ref_subdir   + self.data[ix].split('_')[0] + self.ref_suffix)
            bin = Image.open(self.data_dir + self.bin_subdir + self.data[ix].split('_')[0] + self.bin_suffix)
            # skel = Image.open(self.data_dir + self.skel_subdir + self.data[ix].split('_')[0] + self.skel_suffix)
            # mnt_map = Image.open(self.data_dir + self.mnt_map_subdir + self.data[ix].split('_')[0] + self.mnt_map_suffix)


        # normalizing lat and ref to -1, 1

        lat = transforms.ToTensor()(lat)
        ref = transforms.ToTensor()(ref)


        lat_mean = torch.mean(lat)
        lat_std  = torch.std(lat)

        lat = transforms.Normalize(mean=[lat_mean], std=[2 * lat_std])(lat)


        ref_mean = torch.mean(ref)
        ref_std  = torch.std(ref)


        ref = transforms.Normalize(mean=[ref_mean], std=[2 * ref_std])(ref)

        # if self.skel_transform:
        bin = self.skel_transform(bin)
        mask  = self.skel_transform(mask)
            # mnt_map = self.skel_transform(mnt_map)

        ref_white = ref.max()
        bin_white = bin.max()
        
        ref   = torch.where(mask == 0, ref_white, ref)
        bin = torch.where(mask == 0, bin_white, bin)

        return lat, torch.concat([ref, bin], axis=0)

    def __len__(self):
        return len(self.data)


class PatchEnhancerTrainDataset(Dataset):
    def __init__(self, 
        data_dir: str = "data/", 
        data_list: str = None, 
        transform=None, 
        skel_transform=None, 
        patch_size=None, 
        lat_subdir = '/latents/', 
        ref_subdir = '/references/', 
        skel_subdir = '/skel/', 
        bin_subdir = '/bin/', 
        mask_subdir = '/masks/', 
        mnt_map_subdir='mnt_map', 
        apply_mask = 0, 
        patch_shape = (128,128), 
        stride = 128
    ):
        self.input_row = patch_shape[0]
        self.input_col = patch_shape[1]


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
        self.bin_suffix = "." + os.listdir(data_dir + bin_subdir)[0].split(".")[-1]
        self.mnt_map_suffix = "." + os.listdir(data_dir + mnt_map_subdir)[0].split(".")[-1]
        self.mask_suffix  = "." + os.listdir(data_dir + mask_subdir)[0].split(".")[-1]
        
        self.lat_subdir   = lat_subdir
        self.ref_subdir   = ref_subdir
        self.skel_subdir = skel_subdir
        self.bin_subdir = bin_subdir
        self.mask_subdir  = mask_subdir
        self.mnt_map_subdir = mnt_map_subdir

        self.apply_mask = apply_mask

        self.patch_size = patch_size

        sample_latent = Image.open(self.data_dir+self.lat_subdir+self.data[0]+self.lat_suffix)

        shape_latent = sample_latent.size
        ROW = shape_latent[1]
        COL = shape_latent[0]
        row_list_1 = range(self.input_row, ROW+1, stride)
        row_list_2 = range(ROW, row_list_1[-1]-1,-stride)
        row_list = [*row_list_1, *row_list_2]
        
        col_list_1 = range(self.input_col, COL+1, stride)
        col_list_2 = range(COL, col_list_1[-1]-1, -stride)
        col_list = [*col_list_1,*col_list_2]
        
        self.num_patch = len(self.data) * len(row_list)*len(col_list)

        row_col_inds = np.zeros([self.num_patch,2]).astype(np.int32)


        self.patches_dict = {}
        patch_ind = 0
        for img_i in range(len(self.data)):
            for row_ind in row_list:
                for col_ind in col_list:
                    row_col_inds[patch_ind,:] = [row_ind,col_ind]
                    self.patches_dict[patch_ind] = (self.data[img_i], (row_ind, col_ind))
                    patch_ind += 1


    def __len__(self):
        return self.num_patch

    def __getitem__(self, ix):

        row_col = self.patches_dict[ix][1]
        row = row_col[0]
        col = row_col[1]

        lat   = Image.open(self.data_dir+self.lat_subdir+self.patches_dict[ix][0]+self.lat_suffix)
        mask  = Image.open(self.data_dir + self.mask_subdir  + self.patches_dict[ix][0] + self.mask_suffix)

        try:
            ref   = Image.open(self.data_dir + self.ref_subdir   + self.patches_dict[ix][0] + self.ref_suffix)
            bin = Image.open(self.data_dir + self.bin_subdir + self.patches_dict[ix][0] + self.bin_suffix)
            # skel = Image.open(self.data_dir + self.skel_subdir + self.patches_dict[ix][0] + self.skel_suffix)
            # mnt_map = Image.open(self.data_dir + self.mnt_map_subdir + self.patches_dict[ix][0] + self.mnt_map_suffix)


        except FileNotFoundError: # especial case when are dealing with an synthetic augmented dataset
            ref   = Image.open(self.data_dir + self.ref_subdir   + self.patches_dict[ix][0].split('_')[0] + self.ref_suffix)
            bin = Image.open(self.data_dir + self.bin_subdir + self.patches_dict[ix][0].split('_')[0] + self.bin_suffix)
            # skel = Image.open(self.data_dir + self.skel_subdir + self.patches_dict[ix][0].split('_')[0] + self.skel_suffix)
            # mnt_map = Image.open(self.data_dir + self.mnt_map_subdir + self.patches_dict[ix][0].split('_')[0] + self.mnt_map_suffix)


        # normalizing lat and ref to -1, 1

        lat = transforms.ToTensor()(lat)
        ref = transforms.ToTensor()(ref)
        bin = self.skel_transform(bin)
        mask  = self.skel_transform(mask)


        # crop images to patch size and coords
        lat = lat[:,(row-self.input_row):row,(col-self.input_col):col]
        ref = ref[:,(row-self.input_row):row,(col-self.input_col):col]
        bin = bin[:,(row-self.input_row):row,(col-self.input_col):col]
        mask = mask[:,(row-self.input_row):row,(col-self.input_col):col]


        lat_mean = torch.mean(lat)
        lat_std  = torch.std(lat)

        epsilon = 1e-6  # or 1e-8 for smaller effect
        lat = transforms.Normalize(mean=[lat_mean], std=[2 * lat_std + epsilon])(lat)


        ref_mean = torch.mean(ref)
        ref_std  = torch.std(ref)


        ref = transforms.Normalize(mean=[ref_mean], std=[2 * ref_std + epsilon])(ref)

        # if self.skel_transform:
            # mnt_map = self.skel_transform(mnt_map)

        ref_white = ref.max()
        bin_white = bin.max()
        
        # ref   = torch.where(mask == 0, ref_white, ref)
        # bin = torch.where(mask == 0, bin_white, bin)


        return lat, torch.concat([ref, bin], axis=0)

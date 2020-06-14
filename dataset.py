import torchvision.transforms as transforms
import os, random
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image, PngImagePlugin

from utils import random_crop, square_patch, custom_bicubic

class COCO(Dataset):

    def __init__(self, image_paths, target_paths, scale_factor=4, patch_size=32):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transforms.ToTensor()
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.upsample_size = self.patch_size * self.scale_factor
        self.PER_CHANNEL_MEANS = np.array([0.47614917, 0.45001204, 0.40904046])
        self.train_imgs = os.listdir(self.image_paths)
        self.target_imgs = os.listdir(self.target_paths)

    def __getitem__(self, index):
        PngImagePlugin.MAX_TEXT_CHUNK = 1000*(1024**2)
        """image, target = random_crop(Image.open(self.image_paths + os.listdir(self.image_paths)[index]),
                                    Image.open(self.target_paths + os.listdir(self.target_paths)[index]),
                                    image_max_range=32,
                                    target_scale=4)"""
        image = square_patch(self.image_paths + self.train_imgs[index], self.patch_size)
        target = square_patch(self.target_paths + self.target_imgs[index], self.patch_size * self.scale_factor)
        patches = []
        patches_target = []
        bicubic_res = []
        for el in image:
            patches.append(torch.from_numpy((np.asarray(el, dtype=np.float64) / 255) - self.PER_CHANNEL_MEANS).view((1, 3, self.patch_size, self.patch_size)))
        patches = torch.cat(patches, dim=0)
        for el in target:
            patches_target.append(torch.from_numpy(np.asarray(el, dtype=np.float64) / 255).view((1, 3, self.upsample_size, self.upsample_size)))
        patches_target = torch.cat(patches_target, dim=0)
        for el in image:
            bicubic_res.append(torch.from_numpy((np.asarray(el.resize((self.upsample_size, self.upsample_size), Image.BICUBIC), dtype=np.float64) / 255)
                                               + self.PER_CHANNEL_MEANS).view((1, 3, self.upsample_size, self.upsample_size)))
        bicubic_res = torch.cat(bicubic_res, dim=0)

        return patches, patches_target, bicubic_res

    def __len__(self):  # return count of sample we have

        return len(self.train_imgs)

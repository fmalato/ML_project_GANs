import torchvision.transforms as transforms
import os, random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from PIL import Image, PngImagePlugin
from skimage import io
from skimage.transform import resize

from utils import random_crop, square_patch, custom_bicubic

class COCO(Dataset):

    def __init__(self, image_paths, target_paths, bicubs_path, scale_factor=4, patch_size=32):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.bicubs_path = bicubs_path
        self.transforms = transforms.ToTensor()
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.upsample_size = self.patch_size * self.scale_factor
        self.train_imgs = os.listdir(self.image_paths)

        PngImagePlugin.MAX_TEXT_CHUNK = 1000 * (1024 ** 2)

    def __getitem__(self, index):

        image = torch.load(self.image_paths + self.train_imgs[index])
        target = torch.load(self.target_paths + self.train_imgs[index])
        bicub = torch.load(self.bicubs_path + self.train_imgs[index])

        return image, target, bicub

    def __len__(self):  # return count of sample we have

        return len(self.train_imgs)

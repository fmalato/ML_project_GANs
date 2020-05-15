import torchvision.transforms as transforms
import os, random

from torch.utils.data import Dataset
from PIL import Image

from utils import random_crop, square_patch

class COCO(Dataset):

    def __init__(self, image_paths, target_paths):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):

        """image, target = random_crop(Image.open(self.image_paths + os.listdir(self.image_paths)[index]),
                                    Image.open(self.target_paths + os.listdir(self.target_paths)[index]),
                                    image_max_range=32,
                                    target_scale=4)"""
        #image = Image.open(self.image_paths + os.listdir(self.image_paths)[index])
        #target = Image.open(self.target_paths + os.listdir(self.target_paths)[index])
        p = random.randint(0, 3)
        image = square_patch(self.image_paths + os.listdir(self.image_paths)[index], 32)[p]
        target = square_patch(self.target_paths + os.listdir(self.target_paths)[index], 128)[p]
        t_image = self.transforms(image)
        t_target = self.transforms(target)
        return t_image, t_target

    def __len__(self):  # return count of sample we have

        return len(os.listdir(self.image_paths))

import random
import os, re

from math import floor
from PIL import Image

# Given that I have to perform the same random crop on both the image and the target, I had to redefine the function
def random_crop(image, target, image_max_range=32, target_scale=4):
    crop_x = random.randint(0, image_max_range)
    crop_y = random.randint(0, image_max_range)
    image = image.crop((crop_x,
                       crop_y,
                       crop_x + image_max_range,
                       crop_y + image_max_range
                        ))
    target = target.crop((crop_x * target_scale,
                         crop_y * target_scale,
                         (crop_x + image_max_range) * target_scale,
                         (crop_y + image_max_range) * target_scale
                          ))
    return image, target

def generate_dataset(src, dst, scale=4):
    imgs = os.listdir(src)
    os.mkdir(dst + '/lr/')
    os.mkdir(dst + '/hr/')

    for i in range(len(imgs)):
        orig = Image.open(src + '/{x}'.format(x=imgs[i]))
        img = crop_central_square(orig)
        i_hr = img.resize((256, 256), Image.ANTIALIAS)
        i_lr = img.resize((int(256 / scale), int(256 / scale)))
        i_hr.save(dst + '/hr/{x}'.format(x=imgs[i]), 'PNG')
        i_lr.save(dst + '/lr/{x}'.format(x=imgs[i]), 'PNG')

def short_side(img):
    w, h = img.size
    if w < h:
        return w
    else:
        return h

def crop_central_square(img):
    w, h = img.size
    if w < h:
        return img.crop((0, floor((h - w) / 2), w, floor(((h - w) / 2) + w)))
    else:
        return img.crop((floor((w - h) / 2), 0, floor(((w - h) / 2) + h), h))
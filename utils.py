import random

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

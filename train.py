import numpy as np

from nets import FCNN
from PIL import Image
from math import floor
from matplotlib import cm

def PixelWiseMSELoss(input, target):
    # 1/whc * sum(w,h,c)(Iestwhc - Ihrwhc)**2
    out = (1 / input.shape[0] * input.shape[1] * input.shape[2]) * ((input - target)**2).sum(0).sum(1).sum(2)
    return out

def loadimg(fn, scale=4):
    try:
        img = Image.open(fn).convert('RGB')
    except IOError:
        return None
    w, h = img.size
    img.crop((0, 0, floor(w/scale), floor(h/scale)))
    img = img.resize((w//scale, h//scale), Image.ANTIALIAS)
    return np.array(img)/255


if __name__ == '__main__':
    #net = FCNN(input_channels=3)
    orig = Image.open('train2014/COCO_train2014_000000000078.jpg')
    width, height = orig.size  # Get dimensions
    new_width = 256
    new_height = 256

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    im = orig.crop((left, top, right, bottom))
    print(orig.size)
    # Resizing the image
    img = loadimg('train2014/COCO_train2014_000000000078.jpg', scale=4)
    img = Image.fromarray(np.uint8(img*255))
    # Testing
    orig.show()
    im.show()
    img.show()
    print(img.size)



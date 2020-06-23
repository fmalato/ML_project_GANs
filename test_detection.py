import numpy as np
import math
import matplotlib.pyplot as plt
import random
import torch

import torchvision.models as models

from skimage.transform import resize
from nets import FCNN


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def reconstruct_images(data_dict, num_images=50):
    d = data_dict[b'data']
    l = data_dict[b'fine_labels']

    idxs = []
    for i in range(num_images):
        # 10000 images in the batch
        idxs.append(random.randint(0, 9999))
    d = [d[i] for i in idxs]
    l = [l[i] for i in idxs]

    num_imgs = len(d)
    img_size = int(math.sqrt(len(d[0]) / 3))
    img = np.zeros((img_size, img_size, 3))
    imgs = []
    labels = []
    idx = 0
    for (image, label) in zip(d, l):
        i, j, k = 0, 0, 0
        for el in image:
            img[j % img_size][i % img_size][k] = el / 255
            i += 1
            if i == 32:
                i = 0
                j += 1
            if j == 32:
                j = 0
                k += 1
        imgs.append(resize(img, (56, 56)))
        labels.append(label)
        img = np.zeros((img_size, img_size, 3))
        idx += 1
        if idx % 50 == 0:
            print('Processed: %d / %d' % (idx, num_images))

    return imgs, labels


"""
    Slightly different test since ImageNet didn't give any authorization to download:
    
    - All test images are updated with bicubic interpolation (baseline) to 224x224 and classified, in order
      to estabilish a baseline. Since labels from ImageNet are different than CIFAR-100, we use 50 images and
      perform a manual correction in case of false positive (i.e. ResNet classifies correctly but label
      doesn't match with CIFAR-100)
    - Then, the same 50 images are SR'd with each model and classified again. As for the baseline, false 
      positives are corrected manually
    - Hopefully, ENet-PAT shows the best results.
"""

# Configuration
num_samples = 50
resnet = models.resnet50(pretrained=True)
enet = FCNN()
weights = 'trained_models/ENet-E.pth'
enet.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
resnet.eval()
enet.eval()
# Getting data
data = unpickle('data_detection/test')
label_names = unpickle('data_detection/meta')
label_names = label_names[b'fine_label_names']
imgs, labels = reconstruct_images(data, num_images=num_samples)
# Bicubic upsampling
imgs = [resize(image, (224, 224)) for image in imgs]
labels_bicub = []







"""idxs = []
for i in range(4):
    idxs.append(random.randint(0, num_samples))
fig, ax1 = plt.subplots(2, 2)
ax1[0][0].imshow(imgs[idxs[0]])
ax1[0][1].imshow(imgs[idxs[1]])
ax1[1][0].imshow(imgs[idxs[2]])
ax1[1][1].imshow(imgs[idxs[3]])
ax1[0][0].set_title(label_names[labels[idxs[0]]].decode('utf-8'))
ax1[0][1].set_title(label_names[labels[idxs[1]]].decode('utf-8'))
ax1[1][0].set_title(label_names[labels[idxs[2]]].decode('utf-8'))
ax1[1][1].set_title(label_names[labels[idxs[3]]].decode('utf-8'))
plt.show()"""

import numpy as np
import math
import matplotlib.pyplot as plt
import random
import torch
import cv2
import os

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import skimage.io as io

from skimage.transform import resize, downscale_local_mean, rescale
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from nets import FCNN
from cifar_resnet import cifar_resnet44

from utils import crop_central_square_np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def reconstruct_images(data_dict, num_images=50):
    d = data_dict[b'data']
    l = data_dict[b'fine_labels']

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    idxs = []
    for i in range(num_images):
        # 10000 images in the batch
        idxs.append(random.randint(0, 9999))
    d = [d[i] for i in idxs]
    l = [l[i] for i in idxs]

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
        imgs.append(transf(resize(img, (8, 8))).view(1, 3, 8, 8))
        labels.append(label)
        img = np.zeros((img_size, img_size, 3))
        idx += 1
        if idx % 50 == 0:
            print('Processed: %d / %d' % (idx, num_images))

    return imgs, labels


def get_prediction(image, threshold):
    transf = transforms.ToTensor()
    img = transf(image)   # Apply the transform to the image
    pred = fastr([img.float()])    # Pass the image to the model
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]    # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]    # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]    # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def object_detection_api(img, threshold=0.5, rect_th=3, text_size=1.5, text_th=3, downscale=False):
    """img = cv2.imread(img_path)  # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB"""
    if downscale:
        text_size /= 4
        rect_th = 1
        text_th = 1
    boxes, pred_cls = get_prediction(img, threshold)  # Get predictions
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0),
                  thickness=rect_th)  # Draw Rectangle with the coordinates
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                thickness=text_th)  # Write the prediction class
    #print('Classes for {x}: {y}'.format(x=img_path, y=pred_cls))
    plt.figure()  # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()


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
fastr = fasterrcnn_resnet50_fpn(pretrained=True)
fastr.float()
enetE = FCNN()
enetE.float()
weightsE = 'trained_models/ENet-E.pth'
enetE.load_state_dict(torch.load(weightsE, map_location=torch.device('cpu')))
enetPAT = FCNN()
enetPAT.float()
weightsPAT = 'trained_models/ENet-PAT.pth'
enetPAT.load_state_dict(torch.load(weightsPAT, map_location=torch.device('cpu')))
fastr.eval()
enetE.eval()
enetPAT.eval()
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Possible good images to show: 0, 2, 1792!,
coco = os.listdir('evaluation/val2017/')
tens = transforms.ToTensor()
idxs = [random.randint(0, len(coco) - 1) for i in range(2)]
print(idxs)
images_norm = [io.imread('evaluation/val2017/{x}'.format(x=coco[i])) for i in idxs]
images_down = []
print('Downscaling images...')
for img in images_norm:
    h, w, c = img.shape
    images_down.append(np.array(downscale_local_mean(img, (4, 4, 1)), dtype=np.uint8))
images_E = []
print('Upscaling with ENet-E...')
for img in images_down:
    h, w, c = img.shape
    inp = tens(img / 255).float()
    inp = inp.view((1, c, h, w))
    output = enetE(inp)
    o = output.view((c, h * 4, w * 4))
    o = o.data.numpy()
    o = np.swapaxes(o, 0, 1)
    o = np.swapaxes(o, 1, 2)

    bicub_res = rescale(img, (4, 4, 1), anti_aliasing=True)
    result = np.clip(o + bicub_res, 0., 1.) * 255
    images_E.append(np.array(result, dtype=np.uint8))
images_PAT = []
print('Upscaling with ENet-PAT...')
for img in images_down:
    h, w, c = img.shape
    inp = tens(img / 255).float()
    inp = inp.view((1, c, h, w))
    output = enetPAT(inp)
    o = output.view((c, h * 4, w * 4))
    o = o.data.numpy()
    o = np.swapaxes(o, 0, 1)
    o = np.swapaxes(o, 1, 2)

    bicub_res = rescale(img, (4, 4, 1), anti_aliasing=True)
    result = np.clip(o + bicub_res, 0., 1.) * 255
    images_PAT.append(np.array(result, dtype=np.uint8))
print('Testing downscaled images')
for idx in images_down:
    object_detection_api(idx, threshold=0.8, downscale=True)
print('Testing normal images')
for idx in images_norm:
    object_detection_api(idx, threshold=0.8, downscale=False)
print('Testing ENet-E images')
for idx in images_E:
    object_detection_api(idx, threshold=0.8, downscale=False)
print('Testing ENet-PAT images')
for idx in images_PAT:
    object_detection_api(idx, threshold=0.8, downscale=False)

import core
core.init()

import models
model = core.create_object(models.segmentation, 'SegForestNet', input_shape=[4,224,224], num_classes=8, region_encoder=None)

# from torchsummary import summary
# summary(model)

import torch
import numpy as np
import PIL.Image

file_path   = './tmp/Greenspace/School_50m/'
result_path = './tmp/Greenspace/result/'
lut = (
    (255,255,255), # void
    ( 38, 38, 38), # impervious surface
    (238,118, 33), # building
    ( 34,139, 34), # low vegetation
    (  0,222,137), # high vegetation
    (255,  0,  0), # car
    (  0,  0,238), # water
    (160, 30,230)  # sports venues
)

def preprocess(filename):
    weights = torch.load('./models/pretrain/064_36_4x224x224.pt', map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    img = PIL.Image.open(file_path + filename)
    img = img.resize((224, 224))
    img.save(result_path + filename)
    return img

def segment(img):
    origin = np.asarray(img)
    x = np.empty((4, *origin.shape[:2]))
    for i in range(origin.shape[0]):
        for j in range(origin.shape[1]):
            chan = origin[i][j]
            x[0,i,j], x[1,i,j], x[2,i,j], x[3,i,j] = chan[0], chan[1], chan[2], chan[3]
    x = np.array([x])
    y = model(torch.from_numpy(x).float().to(core.device).requires_grad_())
    return y

def saveResult(filename, img, y):
    result_npy = ''.join([*filename.split('.')[:-1], '_pred'])
    result_img = ''.join([*filename.split('.')[:-1], '_mask.', *filename.split('.')[-1:]])
    y = y[0].argmax(0).numpy()
    np.save(result_path + result_npy, y)
    img_np = np.asarray(img)
    y_img = np.zeros((*y.shape[-2:], 4), dtype=np.uint8)
    for i in range(y_img.shape[0]):
        for j in range(y_img.shape[1]):
            y_img[i,j,:] = [*lut[y[i,j]], img_np[i,j,3]]
    y_img = np.clip(y_img, 0, 255, dtype=np.uint8)
    PIL.Image.fromarray(y_img).save(result_path + result_img)
    print(result_npy)

import os
all_files = os.listdir('./tmp/Greenspace/School_50m/')

for e in all_files:
    img = preprocess(e)
    y = segment(img)
    saveResult(e, img, y)

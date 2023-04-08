import core
core.init()

import models

model = core.create_object(models.segmentation, 'SegForestNet', input_shape=[4,224,224], num_classes=8, region_encoder=None)

# from torchsummary import summary
# summary(model)

filename = 'Baldwin_Early_Learning_Pi.png'
result_name = ''.join([*filename.split('.')[:-1], '_pred.', *filename.split('.')[-1:]])

file_path   = './tmp/Greenspace/School_50m/'
result_path = './tmp/Greenspace/result/'

import torch
weights = torch.load('./models/pretrain/132_model_4x224x224.pt', map_location=torch.device('cpu'))
model.load_state_dict(weights)

import PIL.Image
img = PIL.Image.open(file_path + filename)
img = img.resize((224, 224))
img.save(result_path + filename)

import numpy as np
origin = np.asarray(img)
x = np.empty((4, *origin.shape[:2]))
for i in range(origin.shape[0]):
    for j in range(origin.shape[1]):
        chan = origin[i][j]
        x[0,i,j], x[1,i,j], x[2,i,j], x[3,i,j] = chan[0], chan[1], chan[2], chan[3]
x = np.array([x])
print(x.shape)

y = model(torch.from_numpy(x).float().to(core.device).requires_grad_())
print(y.shape)

lut = (
    (255,255,255), # void
    ( 38, 38, 38), # impervious surface
    (238,118, 33), # building
    ( 34,139, 34), # pervious surface
    (  0,222,137), # high vegetation
    (255,  0,  0), # car
    (  0,  0,238), # water
    (160, 30,230)  # sports venues
)

y = y[0].argmax(0).cpu().numpy()
y_img = np.zeros((*y.shape[-2:], 4), dtype=np.uint8)
for i in range(y_img.shape[0]):
    for j in range(y_img.shape[1]):
        y_img[i,j,:] = [*lut[y[i,j]], origin[i,j,3]]
y_img = np.clip(y_img, 0, 255, dtype=np.uint8)
PIL.Image.fromarray(y_img).save(result_path + result_name)

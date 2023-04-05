import core
core.init()

import models

model = core.create_object(models.segmentation, 'SegForestNet', input_shape=[5,224,224], num_classes=8, region_encoder=None)

# print(model)
from torchsummary import summary
summary(model)

import torch
model.load_state_dict(torch.load('./models/pretrain/104_model.pt'))


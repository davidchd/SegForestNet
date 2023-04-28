# preprocess images to match the color space of the training data

base_img = './tmp/SemCity/img_multispec_05/TLS_BDSD_RGB_noGeo/TLS_BDSD_RGB_noGeo_04.tif'
process_path = './tmp/Greenspace/Crop_50m/'

# compute stats
import PIL.Image
import numpy as np

def computeColorMean(filename):
    img = PIL.Image.open(filename)
    x = np.array(img).transpose(2, 0, 1)
    return [x[i].mean() for i in range(3)]

import os
all_files = os.listdir(process_path)
all_stats = [0, 0, 0]
for f in all_files:
    stat = computeColorMean(process_path + f)
    all_stats = [all_stats[i] + stat[i] for i in range(3)]
all_stats = [all_stats[i] / len(all_files) for i in range(3)]

print(all_stats)

baseline = computeColorMean(base_img)
diff = [baseline[i] - all_stats[i] for i in range(3)]
scalars = [baseline[i] / all_stats[i] for i in range(3)]
print(diff)
print(scalars)

# manually correction
from math import sqrt
scalars[0] = sqrt(scalars[0])
scalars[0] = sqrt(scalars[0])

for f in all_files:
    img = PIL.Image.open(process_path + f)
    x = np.array(img)
    for i in range(3):
        x[:,:,i] = np.clip(x[:,:,i] * scalars[i], 0, 256)
    PIL.Image.fromarray(x).save(process_path + f)

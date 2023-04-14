
filename = 'Quincy,_Josiah_Upper_Scho.png'

result_npy = ''.join([*filename.split('.')[:-1], '_pred.npy'])
result_path = './tmp/Greenspace/result/'

notation = ["void","impervious_surface","building","low_vegetation","tree","car","water","sports_venues",]

import PIL.Image
img = PIL.Image.open(result_path + filename)

import numpy as np
origin = np.asarray(img)[:,:,3] / 255
pred   = np.load(result_path + result_npy)
print(filename + ":")

origin = origin.flatten()
pred   = pred.flatten()

total_pixel = origin.sum()
count_pixel = [0] * len(notation)
for i in range(len(notation)):
    count_pixel[i] += origin[np.where(pred == i)].sum()

for i in range(len(notation)):
    print("%20s:%6.2f%%" % (notation[i], count_pixel[i] * 100.0 / total_pixel))

filename = 'Quincy,_Josiah_Upper_Scho.png'
result_name = ''.join([*filename.split('.')[:-1], '_pred.', *filename.split('.')[-1:]])

file_path   = './tmp/Greenspace/School_50m/'
result_path = './tmp/Greenspace/result/'

import PIL.Image
img = PIL.Image.open(file_path + filename)
img = img.resize((224, 224))

import numpy as np
x = np.asarray(img)
print(x.shape)
x_alpha = x[:,:,3]
print(x_alpha.shape)
print(np.unique(x_alpha))

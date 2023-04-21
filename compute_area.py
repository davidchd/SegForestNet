import math
import PIL.Image
import numpy as np

notation = ["void","impervious_surface","building","low_vegetation","tree","car","water","sports_venues",]

def calculate(filename, result_path='./tmp/Greenspace/result/'):
    result_npy = ''.join([*filename.split('.')[:-1], '_pred.npy'])
    img = PIL.Image.open(result_path + filename)
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
        count_pixel[i] = count_pixel[i] / total_pixel
        # print("%20s:%6.2f%%" % (notation[i], count_pixel[i]))
    return count_pixel

# calculate(filename)

import os
all_files = os.listdir('./tmp/Greenspace/School_50m/')

import csv
with open('school_50m_seg.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["School_Name", 
                     "Area_50m_Concrete_percent", "Area_50m_Concrete_sqm", 
                     "Area_50m_Building_percent", "Area_50m_Building_sqm", 
                     "Area_50m_LowVege_percent", "Area_50m_LowVege_sqm", 
                     "Area_50m_Tree_percent", "Area_50m_Tree_sqm", 
                     "Area_50m_Water_percent", "Area_50m_Water_sqm", ])
    for e in all_files:
        ratio = calculate(e)
        row = [''.join(e.split('.')[:-1]), ]
        for i in [1, 2, 3, 4, 6, ]:
            row.append(ratio[i] * 100.0)
            row.append(ratio[i] * 50 ** 2 * math.pi)
        writer.writerow(row)

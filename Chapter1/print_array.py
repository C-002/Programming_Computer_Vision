from PIL import Image
import numpy as np

im = np.array(Image.open('../data/empire.jpg'))
print(im.shape, im.dtype)

im = np.array(Image.open('../data/empire.jpg').convert('L'), 'f') #'f' cmd trans data type as float
print(im.shape, im.dtype)

print(im[200,100])
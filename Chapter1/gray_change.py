from PIL import Image
import numpy as np
from pylab import *

im = array(Image.open('../data/empire.jpg'))
print(im.shape, im.dtype)
figure(figsize=(16,4))
subplot(141)
imshow(im)
print(im.min(), im.max())
im2 = 255 - im
subplot(142)
imshow(im2)
print(im2.min(), im2.max())
im3 = (100.0/255.0) * im + 100
subplot(143)
imshow(im3.astype('uint8'))
print(im3.min(), im3.max())
im4 = 255.0 * (im/255.0)**2
subplot(144)
imshow(im4.astype('uint8'))
print(im4.min(), im4.max())

show()



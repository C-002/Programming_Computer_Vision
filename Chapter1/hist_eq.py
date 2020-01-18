from PIL import Image
from numpy import *
import imtool as imtools
import matplotlib.pyplot as plt

im = array(Image.open('../data/AquaTermi_lowcontrast.JPG').convert('L'))
im2, cdf = imtools.histeq(im)

plt.figure(figsize=(15,15))
plt.gray()
plt.subplot(231)
plt.hist(im.flatten(), 255, density=True)
plt.xticks([])
plt.yticks([])
plt.subplot(234)
plt.imshow(im)
plt.xticks([])
plt.yticks([])
plt.subplot(232)
plt.plot(cdf)
plt.xticks([])
plt.yticks([])
plt.subplot(233)
plt.hist(im2.flatten(), 255, density=True)
plt.xticks([])
plt.yticks([])
plt.subplot(236)
plt.imshow(im2)
plt.xticks([])
plt.yticks([])
plt.show()
from PIL import Image
import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt

im = np.array(Image.open('../data/empire.jpg').convert('L'))

# sobel导数滤波器
imx = np.zeros(im.shape)
filters.sobel(im, 1, imx)

imy = np.zeros(im.shape)
filters.sobel(im, 0, imy)
# This is my trick with axis: just add the operation in your mind to make it sound clear:
# axis 0 = rows
# axis 1 = columns

magnitude = np.sqrt(imx**2 + imy**2)

plt.figure()
plt.gray()
plt.subplot(1, 4, 1)
plt.imshow(im)
plt.subplot(1, 4, 2)
plt.imshow(imx)
plt.subplot(1, 4, 3)
plt.imshow(imy)
plt.subplot(1, 4, 4)
plt.imshow(magnitude)
plt.show()

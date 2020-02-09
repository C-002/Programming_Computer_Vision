from PIL import Image
import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt

im = np.array(Image.open('../data/empire.jpg').convert('L'))
im2 = filters.gaussian_filter(im, 5)

plt.figure()
plt.gray()
plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
plt.imshow(im2)
plt.show()
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import rof
from scipy.ndimage import gaussian_filter

im = np.array(Image.open('../data/empire.jpg').convert('L'))
im2 = gaussian_filter(im, 5)
U, T = rof.denoise(im, im)

plt.figure(figsize=(20, 5))
plt.gray()
plt.subplot(1, 4, 1)
plt.imshow(im)
plt.subplot(1, 4, 2)
plt.imshow(im2)
plt.subplot(1, 4, 3)
plt.imshow(U)
plt.subplot(1, 4, 4)
plt.imshow(T)
plt.show()

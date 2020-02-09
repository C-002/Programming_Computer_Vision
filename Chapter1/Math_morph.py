from PIL import Image
from scipy.ndimage import measurements, morphology
import numpy as np
import matplotlib.pyplot as plt

im = np.array(Image.open('../data/houses.png').convert('L'))
im = 1*(im < 128)

labels, nbr_objects = measurements.label(im)

print("Number of objects:", nbr_objects)

plt.figure()
plt.gray()
plt.subplot(2, 2, 1)
plt.imshow(im)
plt.subplot(2, 2, 2)
plt.imshow(labels)

im_open = morphology.binary_opening(im, np.ones((9, 5)), iterations=2)
labels_open, nbr_objects_open = measurements.label(im_open)
print("Number of objects through binary open:", nbr_objects_open)
plt.subplot(2, 2, 3)
plt.imshow(im_open)
plt.subplot(2, 2, 4)
plt.imshow(labels_open)

plt.show()
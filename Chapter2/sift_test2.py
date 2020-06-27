import sift
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im1name = 'crans_1_small.jpg'
im1sift = im1name + '.sift'
im1 = np.array(Image.open(im1name).convert('L'))
sift.process_image(im1name, im1sift)
l1, d1 = sift.read_features_from_file(im1sift)
print(l1.shape, d1.shape)

im2name = 'crans_2_small.jpg'
im2sift = im2name + '.sift'
im2 = np.array(Image.open(im2name).convert('L'))
sift.process_image(im2name, im2sift)
l2, d2 = sift.read_features_from_file(im2sift)
print(l2.shape, d2.shape)

print("start matching..")
matches = sift.match_twosided(d1, d2)


plt.figure()
plt.gray()
sift.plot_matches(im1, im2, l1, l2, matches)
plt.show()
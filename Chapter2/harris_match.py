import harris
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

wid = 5

im1 = np.array(Image.open('data/crans_1_small.jpg').convert('L'))
harrisim = harris.compute_harris_response(im1, 5)
filtered_coords1 = harris.get_harris_points(harrisim, wid+1)
d1 = harris.get_descriptors(im1, filtered_coords1, wid)

im2 = np.array(Image.open('data/crans_2_small.jpg').convert('L'))
harrisim = harris.compute_harris_response(im2, 5)
filtered_coords2 = harris.get_harris_points(harrisim, wid+1)
d2 = harris.get_descriptors(im2, filtered_coords2, wid)

print('start matching')
matches = harris.match_twosided(d1, d2)

plt.figure()
plt.gray()
harris.plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches)
plt.show()

from PIL import Image
import numpy as np
import os
import harris


lifePath = os.path.join('D:/WorkSpace/Programming_Computer_Vision/data/', 'empire.jpg')
im = np.array(Image.open(lifePath).convert('L'))
harrisim = harris.compute_harris_response(im)
filtered_coords = harris.get_harris_points(harrisim, 6, 0.05)
harris.plot_harris_points(im, filtered_coords, 1)

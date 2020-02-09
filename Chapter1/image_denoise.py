import numpy as np
from scipy.ndimage import filters
#from scipy.misc import imsave
import matplotlib.pyplot as plt
import rof

#使用噪声创建合成图像
im = np.zeros((500, 500))
im[100:400,100:400] = 128
im[200:300,200:300] = 255
im = im + 30 * np.random.standard_normal((500, 500))

U, T = rof.denoise(im, im)
G = filters.gaussian_filter(im, 10)

plt.imsave('synth_orgin.pdf', im)
plt.imsave('synth_rof.pdf', U)
plt.imsave('synth_gaussian.pdf', G)
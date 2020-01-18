import os
from PIL import Image
from numpy import *

def get_imlist(path):
    """ return all JPG image filenames in the path """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


def imresize(im, sz):
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))

def histeq(im, nbr_bins=256):
    """ 对一副灰度图像进行直方图均匀化"""

    # 计算直方图
    imhist, bins = histogram(im.flatten(), nbr_bins, density=True)
    cdf = imhist.cumsum() # 累计分布函数
    #print(cdf)
    cdf = 255 * cdf / cdf[-1]

    # 使用累计分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

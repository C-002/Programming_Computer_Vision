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
    #print(len(imhist))
    cdf = imhist.cumsum() # 累计分布函数
    #print(cdf)
    cdf = 255 * cdf / cdf[-1]
    #print(bins)
    # 使用累计分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def compute_average(imlist):
    """ 计算图像列表的平均图像 """

    #打开第一幅图像，将其存储在浮点型数组中
    averageim = array(Image.open(imlist[0]), 'f')

    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname+'...skipped')
        
    averageim /= len(imlist)

    #返回 uint8 类型的平均图像
    return array(averageim, 'uint8')

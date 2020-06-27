# -*- coding: utf-8 -*-
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt


def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    """ 处理一幅图片，然后将结果保存在文件中 """
    if imagename[-3:] != 'pgm':
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    cmmd = str("sift " + imagename + " --output=" + resultname + " " + params)
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)


def read_features_from_file(filename):
    f = np.loadtxt(filename)
    return f[:, :4], f[:, 4:]

def write_features_to_file(filename, locs, desc):
    np.savetxt(filename, np.hstack((locs, desc)))

def plot_features(im, locs, circle=False):

    def draw_circle(c, r):
        t = np.arange(0, 1.01, .01)*2*np.pi
        x = r*np.cos(t) + c[0]
        y = r*np.sin(t) + c[1]
        plt.plot(x, y, 'b', linewidth=2)

    plt.imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plt.plot(locs[:,0], locs[:,1],'ob')

    plt.axis('off')

def match(desc1, desc2):
    """ 对于第一幅图像中的每个描述子，选取其在第二幅图像中的匹配"""
    """ 输入: desc1(第一幅图像中的描述子), desc2(第二幅图像中的描述子)"""

    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape
    
    matchscores = np.zeros((desc1_size[0]), 'int')
    desc2t = desc2.T #预先计算矩阵转置
    
    for i in range(desc1_size[0]):
        #余弦相似性
        dotprods = np.dot(desc1[i,:],desc2t)
        dotprods = 0.9999*dotprods
        #反余弦和反排序，返回第二幅图像中特征的索引
        index = np.argsort(np.arccos(dotprods))
         
        if  np.arccos(dotprods)[index[0]] < dist_ratio * np.arccos(dotprods)[index[1]]:
            matchscores[i] = int(index[0])
         
        #print(matchscores)   
    return matchscores

def match_twosided(desc1, desc2):
    
    print("matching 1 to 2 ...")
    matches_12 = match(desc1, desc2)
    print("matching 2 to 1 ...")
    matches_21 = match(desc2, desc1)
    
    ndx_12 = matches_12.nonzero()[0]
    
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
            
    return matches_12

def appendimages(im1, im2):

    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-row1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows2-row1, im2.shape[1]))), axis=0)

    return np.concatenate((im1, im2), axis=1)

def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):

    im3 = appendimages(im1, im2)
    if show_below:
        im3 = np.vstack((im3, im3))

    plt.imshow(im3)
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            #print(i, m)
            #print([locs1[i][0],locs2[m][0]+cols1],[locs1[i][1],locs2[m][1]])
            plt.plot([locs1[i][0],locs2[m][0]+cols1],[locs1[i][1],locs2[m][1]],'c')
    plt.axis('off')

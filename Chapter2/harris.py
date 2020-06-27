# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt

def compute_harris_response(im, sigma=3):

    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    Wdet = Wxx*Wyy - Wxy**2
    Wtr  = Wxx + Wyy

    return Wdet / Wtr

def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    coords = np.array(harrisim_t.nonzero()).T

    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    index = np.argsort(candidate_values)

    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1

    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0], coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                                (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
            
    return filtered_coords

def plot_harris_points(image, filtered_coords, alpha=0.5):
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords], '*', alpha=alpha, markersize=3)
    plt.axis('off')
    plt.show()


def get_descriptors(image, filtered_coords, wid=5):
	desc = []
	for coords in filtered_coords:
		patch = image[coords[0]-wid:coords[0]+wid+1,
					  coords[1]-wid:coords[1]+wid+1].flatten()			  
		desc.append(patch)
	
	return desc
	
def match(desc1, desc2, threshold=0.5):

    n = len(desc1[0])
    print('matching..')
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1)
            if ncc_value > threshold:
                d[i, j] = ncc_value

    ndx = np.argsort(-d)
    matchscores = ndx[:,0]

    return matchscores

def match_twosided(desc1, desc2, threhold=0.5):

    matches_12 = match(desc1, desc2, threhold)
    matches_21 = match(desc2, desc1, threhold)

    ndx_12 = np.where(matches_12 >= 0)[0]
    print('match_twosided ..')
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

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
            plt.plot([locs1[i][1], locs2[m][1]+cols1], [locs1[i][0], locs2[m][0]], 'c')
    plt.axis('off')

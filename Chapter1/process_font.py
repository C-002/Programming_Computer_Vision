from imtool import *
import pca
import numpy as np
import matplotlib.pyplot as plt
import pickle

font_images_path = '../data/fontimages/a_thumbs/'
selected_font_images_path = '../data/selectedfontimages/a_selected_thumbs/'

imlist = get_imlist(font_images_path)
#print(imlist)

im = np.array(Image.open(imlist[0]))
m, n = im.shape[0:2]
imnbr = len(imlist)
print(m, n, imnbr)

immatrix = np.array([np.array(Image.open(im)).flatten() for im in imlist], 'f')
print(immatrix.shape)

V, S, immean = pca.pca(immatrix)

plt.figure()
plt.gray()
plt.subplot(2, 4, 1)
plt.imshow(immean.reshape(m, n))
for i in range(7):
    plt.subplot(2, 4, i+2)
    plt.imshow(V[i].reshape(m, n))

plt.show()

with open('font_pca_modes.pkl', 'wb') as f:
    pickle.dump(m, f)
    pickle.dump(n, f)
    pickle.dump(immean, f)
    pickle.dump(V, f)



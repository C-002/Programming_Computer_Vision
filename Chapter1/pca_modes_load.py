import pickle
import matplotlib.pyplot as plt

with open('font_pca_modes.pkl', 'rb') as f:
    m = pickle.load(f)
    n = pickle.load(f)
    immean = pickle.load(f)
    V = pickle.load(f)

plt.figure()
plt.gray()
plt.subplot(2, 4, 1)
plt.imshow(immean.reshape(m, n))
for i in range(7):
    plt.subplot(2, 4, i+2)
    plt.imshow(V[i].reshape(m, n))

plt.show()
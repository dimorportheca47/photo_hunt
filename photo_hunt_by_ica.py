import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA


im1 = cv2.imread("./sumplefig/saize_1.jpeg", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("./sumplefig/saize_2.jpeg", cv2.IMREAD_GRAYSCALE)

plt.imshow(im1, cmap="gray"), plt.axis("off")
plt.show()
plt.imshow(im2, cmap="gray"), plt.axis("off")
plt.show()
im1.shape
# (1050, 1040)
im2.shape
# (1050, 1042)
im2 = im2[:, :1040]

im1 = im1[:, :1040-3]
im2 = im2[:, 3:]
yoko, tate = im2.shape
# (yoko, tate)

# flatten
im1 = im1.flatten()
im2 = im2.flatten()

im1.reshape((1, len(im1))).shape
np.concatenate((im1.reshape((1, len(im1))), im2.reshape((1, len(im2)))))
# data = (dimention x num of components)
data = np.concatenate((im1.reshape((1, len(im1))), im2.reshape((1, len(im2))))).T
data.shape

decomposer = FastICA(n_components = 2)

M = np.mean(data, axis = 1)[:,np.newaxis]
#各データから平均を引く.
data2 = data - M

plt.imshow(data2[:, 0].reshape((yoko, tate)), cmap="gray"), plt.axis("off")
plt.show()
plt.imshow(data2[:, 1].reshape((yoko, tate)), cmap="gray"), plt.axis("off")
plt.show()
decomposer.fit(data)
S = decomposer.transform(data)

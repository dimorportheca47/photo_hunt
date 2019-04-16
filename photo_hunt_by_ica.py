import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA


im1 = cv2.imread("./sumplefig/saize_1.jpeg", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("./sumplefig/saize_2.jpeg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("./saize_fig/im1.jpeg", im1)
cv2.imwrite("./saize_fig/im2.jpeg", im2)

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
# data = (dimention x num of components)?
data = np.concatenate((im1.reshape((1, len(im1))), im2.reshape((1, len(im2))))).T
data.shape
# (1092000, 2)

decomposer = FastICA(n_components = 2)

M = np.mean(data, axis = 1)[:,np.newaxis]
#各データから平均を引く.
data2 = data - M
print("平均引いたやつ")
plt.imshow(data2[:, 0].reshape((yoko, tate)), cmap="gray"), plt.axis("off")
plt.show()
plt.imshow(data2[:, 1].reshape((yoko, tate)), cmap="gray"), plt.axis("off")
plt.show()
# del data
decomposer.fit(data)
S = decomposer.transform(data)
del data2

Uica = decomposer.mixing_
S.shape
s1 = S[:, 0]
s2 = S[:, 1]
s1.shape
M = M.reshape((M.shape[0],))
s1 = s1.reshape((yoko, tate))
s2 = s2.reshape((yoko, tate))
plt.figure(figsize=(30, 30))
plt.imshow(s1, cmap="gray"), plt.axis("off")
# plt.savefig("./saize_fig/s1fig.jpeg")
plt.show()
plt.figure(figsize=(30, 30))
plt.imshow(s2, cmap="gray"), plt.axis("off")
# plt.savefig("./saize_fig/s2fig.jpeg")
plt.show()

# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = np.zeros((256, 256))

w = 32
h = 32

for i in range(256 // h):
    mod = i % 2
    for j in range(256 // w):
        if j % 2 + mod == 1:
            img[h * i : h * (i + 1), w * j : w * (j + 1)] = 255

plt.imshow(img)
plt.show()

cv2.imwrite("checkerboard.png", img)

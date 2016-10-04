import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import copy

img = Image.open("http://i.imgur.com/X017qGH.jpg")

imgMatrix = np.array(list(img.getdata(band=0)), float)
imgMatrix.shape = (img.size[1], img.size[0])
imgMatrix = np.matrix(imgMatrix)
plt.imshow(imgMatrix, cmap='gray')
plt.show()

U, sigma, V = np.linalg.svd(imgMatrix)
print sigma

reconstimg = np.matrix(U[:,:1] * np.diag(sigma[:1]) * np.matrix(V[:1, :]))

# Singular values
plt.figure(1)
orderedPlot, = plt.plot(range(100), sigma[:100], 'bD')
plt.title('singular values')
shuffledSigma = copy.deepcopy(sigma)
random.shuffle(shuffledSigma)	
shuffledPlot, = plt.plot(range(100), shuffledSigma[:100], 'r^')
plt.legend((orderedPlot, shuffledPlot), ("100 largest","shuffled"), loc=1)

plt.figure(2)
plt.rcParams.update(plt.rcParamsDefault)

def plotImg(position, k):
	plt.subplot(position)
	reconstimg = np.matrix(U[:,:k] * np.diag(sigma[:k]) * np.matrix(V[:k, :]))
	plt.imshow(reconstimg, cmap='gray')
	plt.title('k = {}'.format(k))

ks = [2, 10, 20]
positions = [222, 223, 224]

plt.subplot(221)
plt.imshow(imgMatrix, cmap='gray')
plt.title('Original')

for i, k in enumerate(ks):
	plotImg(position[i], k)

plt.show()

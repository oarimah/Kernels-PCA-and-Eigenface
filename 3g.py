#Ositadinma Arimah
#3g
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# reshape image to 64x64 px
def reshapeImage(img):
    reshapedImg = img.reshape(64, 64)
    return reshapedImg


# get the image from the file
def getImage(txt, num):
    image = txt[num, :]
    return image


# load and extract the data from the file
rawImage = np.loadtxt('faces.dat.txt').astype(np.int8)

# init. the principal components
principal_components = [10, 100, 200, 399]

for i in range(0, len(rawImage)):
    mean = np.mean(rawImage[i, :])
    rawImage[i, :] = rawImage[i, :] - mean

# fitting pca over the given data
pca = PCA(399)
pca.fit(rawImage)
components = pca.components_

image100 = getImage(rawImage, 99)  # get image 100 from the file
image100_final = reshapeImage(image100)  # reformat image
reconstructedImg = np.zeros((64, 64))  # init. the final reconstructed image

for m in principal_components:
    for n in range(0, m):
        vector = pca.components_[n]
        vector = reshapeImage(np.asarray(vector))
        reconstructedImg = reconstructedImg + ((vector @ vector.T) @ image100_final)
    plt.imshow(np.swapaxes(reconstructedImg, 0, 1))
    plt.title("3g. Reconstructed Images - Principal Component:" + str(m))
    plt.show()

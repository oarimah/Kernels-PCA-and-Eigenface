#Ositadinma Arimah
#3f
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# reshape image to 64x64 px
def reshapeImage(img):
    reshapedImg = img.reshape(64, 64)
    return reshapedImg


# load and extract the data from the file
rawImage = np.loadtxt('faces.dat.txt').astype(np.int8)

for i in range(0, len(rawImage)):
    mean = np.mean(rawImage[i, :])
    rawImage[i, :] = rawImage[i, :] - mean

# fitting pca over the given data
pca = PCA(400)
pca.fit(rawImage)
components = pca.components_

# display the top-5 leading eigenvectors
for i in range(5):
    pcaImage = pca.components_[i]
    finalImage = reshapeImage(np.asarray(pcaImage))
    plt.imshow(np.swapaxes(finalImage, 0, 1))
    plt.title("Top-5 leading Eigenvectors")
    plt.xlabel(str(i+1) + " Ranked Eigenvalue - 400 Total")
    plt.show()

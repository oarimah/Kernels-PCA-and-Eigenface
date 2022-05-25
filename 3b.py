#Ositadinma Arimah
#3b
import numpy as np
import matplotlib.pyplot as plt


# reshape image to 64x64 px
def reshapeImage(img):
    reshapedImg = img.reshape(64, 64)
    return reshapedImg


# get the image from the file
def getImage(txt, num):
    image = txt[num, :]
    return image


# load and extract the data from the file
rawImage = np.loadtxt('faces.dat.txt')

# extract the mean from each column
mean = rawImage.mean(axis=0)
result = rawImage - mean

# Getting the 100th image
img_extracted = getImage(result, 99)

finalImg = reshapeImage(img_extracted) #reshape the image which was extracted

# plot the graph
plt.title("100th Image With Mean Removed")
plot = plt.figure(1)

# display the image
plt.imshow(np.swapaxes(finalImg, 0, 1))
plt.show()

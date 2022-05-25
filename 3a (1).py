#Ositadinma Arimah
#3a
import numpy as np
import matplotlib.pyplot as plt


# reshape image to 64x64 px
def reshapeImage(img):
    reshapedImg = img.reshape(64, 64)
    return reshapedImg


# get the 200th image from the data file
def getImage(txt, num):
    image = txt[num, :]
    return image


# load and extract the data from the file
rawImage = np.loadtxt('faces.dat.txt')

# display the 200th image
data_extracted = getImage(rawImage, 199)

finalImage = reshapeImage(data_extracted)

# plot the graph
plot = plt.figure(1)
plt.title("200th Image")

# display the image
plt.imshow(np.swapaxes(finalImage, 0, 1))
plt.show()

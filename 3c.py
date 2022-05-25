#Ositadinma Arimah
#3c
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load and extract the data from the file
rawImage = np.loadtxt('faces.dat.txt').astype(np.int8)

# create pca object used for the mean-centered data matrix and compute eigenvectors
pca = PCA(400)
pca.fit(rawImage)
component = pca.components_  # compute components

# EIGENVALUES pca.explained_variance_
eigenValues = pca.explained_variance_
plt.title("Eigenvalues In Descending Order")
plt.xlabel("EigenValue # - 400 Total")
plt.ylabel("Values of Eigenvalue")
plt.plot(eigenValues)
plt.show()

# output eigenvalues in descending order
print("Eigenvalues (descending order): ")
sortedValues = component[:, np.argsort(eigenValues)[::-1]]
for i in range(len(eigenValues)):
    eigenArray = [(np.abs(eigenValues[i]), sortedValues[:, i])]

for j in eigenArray:
    print(j[0])

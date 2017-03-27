import numpy as np
import scipy
#from scipy import misc
#im = misc.imread('../Data/images/low_risk_1.jpg')

from skimage import io, filters
mole_img = np.array(io.imread("../Data/images/low_risk_1.jpg"),
              dtype=np.float64)
edges = filters.sobel(mole_img[:,:,2])
io.imshow(edges)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin
#from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 4

# Load the Summer Palace photo

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
mole_img = np.array(mole_img, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(mole_img.shape)
assert d == 3
image_array = np.reshape(mole_img, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))

codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image')
plt.imshow(mole_img)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image kmeans')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image random')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()

import time
from sklearn.feature_extraction import image

# Downsample the image by a factor of 4
#mole_img = mole_img[:,:,1]
#mole_img = mole_img[::2, ::2] + mole_img[1::2, ::2] + mole_img[::2, 1::2] + mole_img[1::2, 1::2]
#mole_img = mole_img[::2, ::2] + mole_img[1::2, ::2] + mole_img[::2, 1::2] + mole_img[1::2, 1::2]
#
mole_img = scipy.misc.imresize(recreate_image(kmeans.cluster_centers_, labels, w, h)[:,:,0], 0.3)
X = np.reshape(mole_img, (-1,1))
# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.grid_to_graph(*mole_img.shape)

# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
#beta = 6
#eps = 1e-6
#graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
#graph.data = np.exp(-beta * graph.data / graph.data.std())

# Apply spectral clustering (this step goes much faster if you have pyamg
# installed)
N_REGIONS = 4

t0 = time.time()
ward = AgglomerativeClustering(n_clusters=N_REGIONS, 
                                 connectivity=graph,
                                 linkage="ward")
ward.fit(X)
t1 = time.time()
labels = ward.labels_.reshape(mole_img.shape)

plt.imshow(mole_img, cmap=plt.cm.gray)
for l in range(N_REGIONS):
    plt.contour(labels == l, contours=1, colors=[plt.cm.spectral(l / float(N_REGIONS)), ])
plt.xticks(())
plt.yticks(())
title = 'Spectral clustering: %.2fs' % (t1 - t0)
plt.title(title)

plt.figure()
cs = plt.contour(mole_img, origin="image")


#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.matlib import repmat
from sklearn.cluster import SpectralClustering
from skimage import data, io, filters

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def get_data_csv(path):
   df = pd.read_csv(path, header=None)
   df = df.drop(100, axis=1)
   arr = df.to_numpy()
   arr = arr.flatten()
   arr = arr.reshape(6900,1)
   return arr.T[0]


# In[4]:


array1 = get_data_csv("Al At.csv")
array2 = get_data_csv("Al_Wt_pct.csv")
array3 = get_data_csv("Ni At.csv")
array4 = get_data_csv("Ni_Wt_pct.csv")


numpy_array = array1, array2, array3, array4
numpy_array = np.transpose(numpy_array)
numpy_array


# In[5]:


sc = SpectralClustering(n_clusters=2, assign_labels='discretize')
image = sc.fit_predict(numpy_array)


# In[6]:


image_reshape = np.reshape(image, (-1, 100))

# ... or any other NumPy array!
edges = filters.sobel(image_reshape)
io.imshow(edges)
io.show()


# In[7]:


from skimage import exposure

#p2, p98 = np.percentile(image_reshape, (2, 98))

img_rescale = exposure.rescale_intensity(image_reshape, in_range=(0, 100))# and then
io.imsave('filename.jpg', img_rescale)


# In[9]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt_image = image_reshape
#plt_image[0] = 2

imgplot = plt.imshow(image_reshape, cmap=cm.Set1)
plt.savefig('example.png')


# In[5]:


numpy_array_atomic = [list(array1), list(array3)]
numpy_array_atomic = np.transpose(numpy_array_atomic)


# In[6]:


sc = SpectralClustering(n_clusters=2, assign_labels='discretize')
image_atomic = sc.fit_predict(numpy_array_atomic)


# In[7]:


image_atomic_reshape = np.reshape(image_atomic, (-1, 100))
# ... or any other NumPy array!
edges = filters.sobel(image_atomic_reshape)
io.imshow(edges)
io.show()


# In[8]:


numpy_array_weight = [list(array2), list(array4)]
numpy_array_weight = np.transpose(numpy_array_weight)


# In[9]:


sc = SpectralClustering(n_clusters=2, assign_labels='discretize')
image_weight = sc.fit_predict(numpy_array_weight)


# In[10]:


image_reshape_weight = np.reshape(image_weight, (-1, 100))
# ... or any other NumPy array!
edges = filters.sobel(image_reshape_weight)
io.imshow(edges)
io.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from sklearn.cluster import SpectralClustering
from skimage import data, io, filters

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def get_data_csv(path):
   df = pd.read_csv(path, header=None)
   df = df.drop(360, axis=1)
   arr = df.to_numpy()
   arr = arr.flatten()
   arr = arr.reshape(89280,1)
   return arr.T[0]


# In[6]:


array1 = get_data_csv("Boron_at_pct.csv")
array2 = get_data_csv("Carbon_at_pct.csv")
array3 = get_data_csv("Nickel_at_pct.csv")
array4 = get_data_csv("Si_at_pct.csv")
array5 = get_data_csv("W_at_pct.csv")

numpy_array = array1, array2, array3, array4, array5
numpy_array = np.transpose(numpy_array)


# In[ ]:


sc = SpectralClustering(n_clusters=5, assign_labels='discretize')
image = sc.fit_predict(numpy_array)


# In[6]:


image_reshape = np.reshape(image, (-1, 360))

# ... or any other NumPy array!
edges = filters.sobel(image_reshape)
io.imshow(edges)
io.show()


# In[9]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt_image = image_reshape
#plt_image[0] = 2

imgplot = plt.imshow(image_reshape)
plt.savefig('example.png')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from sklearn.cluster import SpectralClustering
from skimage import data, io, filters

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def get_data_csv(path):
   df = pd.read_csv(path, header=None)
   df = df.drop(256, axis=1)
   arr = df.to_numpy()
   arr = arr.flatten()
   arr = arr.reshape(45056,1)
   return arr.T[0]


# In[4]:


array1 = get_data_csv("Al_Wt_pct.csv")
array2 = get_data_csv("Cr_Wt_pct.csv")
array3 = get_data_csv("Fe_Wt_pct.csv")
array4 = get_data_csv("Mo_Wt_pct.csv")
array5 = get_data_csv("Ni_Wt_pct.csv")
array6 = get_data_csv("Al_Wt_pct.csv")
array7 = get_data_csv("Cr_Wt_pct.csv")
array8 = get_data_csv("Fe_Wt_pct.csv")
array9 = get_data_csv("Mo_Wt_pct.csv")
array10 = get_data_csv("Ni_Wt_pct.csv")

numpy_array = array1, array2, array3, array4, array5, array6, array7, array8, array9, array10
numpy_array = np.transpose(numpy_array)


# In[ ]:


sc = SpectralClustering(n_clusters=6, assign_labels='discretize')
image = sc.fit_predict(numpy_array)


# In[6]:


image_reshape = np.reshape(image, (-1, 256))

# ... or any other NumPy array!
edges = filters.sobel(image_reshape)
io.imshow(edges)
io.show()


# In[9]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt_image = image_reshape

imgplot = plt.imshow(image_reshape)
plt.savefig('example.png')


# In[ ]:





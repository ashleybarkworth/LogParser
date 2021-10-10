#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
import distance

def affinityPropogation():
    df = pd.read_csv('/Users/chandni/BRICKS/SIEM/Windows_2k.csv')
    words = df['Content'].tolist()
    words =  np.asarray(words) #So that indexing with a list will work
    lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])
    # print(lev_similarity)

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
    affprop.fit(lev_similarity)
    for cluster_id in np.unique(affprop.labels_):
        print(cluster_id)
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    #     print(affprop.cluster_centers_indices_[cluster_id])
        cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
        cluster_str = ", ".join(cluster)
        print(" - %s  *%s \n" % ("Cluster Name:- "+exemplar+"\n", cluster_str))

affinityPropogation()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import mpl_toolkits
from statistics import mean
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import sklearn.utils
os.getcwd()
import pandas_profiling as pp
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from sklearn.preprocessing import normalize 
from sklearn.decomposition import PCA 

from scipy.spatial.distance import pdist, squareform
pd.set_option('display.max_columns', None)


# In[ ]:


df = pd.read_csv(r'C:\Users\fawaz\Desktop\us-accidents\US_Accidents_May19.csv')
df.head()


# In[24]:


df.info(null_counts=True)


# In[4]:


#Working with a reduced dataset
my_tab = pd.crosstab(index = [df["State"], df["Junction"], df["Sunrise_Sunset"]], columns="counts")
my_tab


# In[5]:


#Working with a reduced dataset
df_reduced = df[(df.State == "TX") & (df.Junction) & (df.Severity > 2)]


# In[6]:


#Using folium to plot heatmaps
import folium
from folium.plugins import HeatMap, MarkerCluster, FastMarkerCluster


# In[7]:


#Define a default map object
def generateBaseMap(default_location=[df_reduced['Start_Lat'].mean(), df_reduced['Start_Lng'].mean()], default_zoom_start=7):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start, prefer_canvas = True)
    return base_map

generateBaseMap()


# In[8]:


base_map = generateBaseMap()
mc = MarkerCluster()
#fmc = FastMarkerCluster()

#for row in df_reduced.itertuples():
    #mc.add_child(folium.Marker(location=[row.Start_Lat, row.Start_Lng]))
    #fmc.add_child(folium.Marker(Location = [row.Start_Lat, row.Start_Lng]))
    
base_map.add_child(FastMarkerCluster(df_reduced[['Start_Lat', 'Start_Lng']].values.tolist()))
base_map


# In[9]:


#Add a heat map
accident_array = df_reduced[['Start_Lat', 'Start_Lng']].values
base_map.add_child(HeatMap(accident_array, radius=15))
base_map


# In[10]:


coords = df_reduced[['Start_Lat', 'Start_Lng']].values


# In[11]:


kms_per_radian = 6371.0088
epsilon = 1.5/kms_per_radian
db = DBSCAN(eps = epsilon, min_samples = 200, algorithm = 'ball_tree', metric = 'haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters = pd.array([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))
clusters


# In[98]:


scaler = StandardScaler()
#df_reduced = df_reduced.drop('ID', axis = 1)
df_scaled = StandardScaler().fit_transform(coords)
#plt.scatter(df_reduced[['Start_Lng']], df_reduced[['Start_Lat']])
plt.scatter(clusters[0][:,0], clusters[0][:,1])


# 

# In[13]:


loc_list = list(clusters[2])


# In[14]:


bm2 = generateBaseMap()
for point in range(0, len(loc_list)):
    folium.Marker(loc_list[point]).add_to(bm2)
bm2


# In[99]:


clusters[0][:,1]


# In[103]:


# kms_per_radian = 6371.0088
# epsilon = .000000001/kms_per_radian
# db2 = DBSCAN(eps = epsilon, min_samples = 10, algorithm = 'ball_tree', metric = 'haversine').fit(clusters[2])
# cluster_labels2 = db2.labels_
# num_clusters = len(set(cluster_labels2))
# clusters2 = 
# print('Number of clusters: {}'.format(num_clusters))
# clusters2

kms_per_radian = 6371.0088
epsilon = 1.5/kms_per_radian
db = DBSCAN(eps = epsilon, min_samples = 10, algorithm = 'ball_tree', metric = 'haversine').fit(clusters[2])
cluster_labels = db.labels_
num_clusters = len(set(cluster_labels))
clusters[2] 
clusters2 = pd.array([clusters[2][cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))
clusters2[0]


# loc_list = list(clusters2[3])
# print(loc_list)

# In[106]:


bm3 = generateBaseMap()
for point in range(0, len(loc_list)):
    folium.Marker(loc_list[point]).add_to(bm3)
    
average = list(map(lambda y: sum(y)/len(y), zip(*loc_list)))
#print(average)
folium.Circle([average[0],average[1]],radius = 500).add_to(bm3)

bm3


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[18]:


import xarray as xr
import pandas as pd
path = '/Users/ls/Downloads/data.grib'
data = '/Users/ls/Downloads/ship_data.csv'
ship = pd.read_csv(data,sep=',')
df1 = xr.open_dataset(path)


# In[19]:


WVPI = ship.loc[0,'wvpi'].split(',')
WVPI = [int(WVPI[i]) for i in range(len(WVPI))]
WD_min = ship.loc[0,'wd_min'].split(',')
WD_min = [int(WD_min[i]) for i in range(len(WD_min))]


# In[20]:


df1


# In[39]:


u10 = df1.u10.data[1:100,1:100,1:100]
v10 = df1.v10.data[1:100,1:100,1:100]
time = df1.time.data[1:100]
latitudes = df1.latitude.data[1:100]
longitudes = df1.longitude.data[257:158:-1]


# In[22]:


len(latitudes)


# In[23]:


(pd.DataFrame([latitudes,longitudes,u10]).T).iloc[0:62]


# In[24]:


df1


# In[25]:


import numpy as np


nodes = np.array([[lat, lon] for lat in latitudes for lon in longitudes])
# latitudes = latitudes[:, :100]
# longitudes = longitude[:, :100]
# u10 = u10[:, :100]  # Subset the first 100 nodes, for example
# v10 = v10[:, :100]
# nodes = nodes[:100]
# time = time[:100]



u = u10.reshape(len(time), len(latitudes) * len(longitudes))
v = v10.reshape(len(time), len(latitudes) * len(longitudes))


# In[26]:


import numpy as np
from datetime import datetime
from scipy.interpolate import griddata

from halem import BaseRoadmap, HALEM_space, HALEM_time, plot_timeseries, HALEM_cost,HALEM_co2

import matplotlib.pyplot as plt


# In[27]:


import numpy as np
from halem import BaseRoadmap

class CustomRoadmap(BaseRoadmap):
    def __init__(
        self,
        u10,  
        v10,  
        time, 
        nodes,  
        number_of_neighbor_layers=1,  
        vship=np.array([[10, 40, 80], [10, 50, 100]]),
        WD_min=WD_min,
        WVPI=WVPI,
        WWL=data.wvpi,
        LWL=data.lwl,
        ukc=data.ukc,
        nl=(data.nl_c,data.nl_m),
        blend=data.blend,
        nodes_index=None,
        *args,
        **kwargs
    ):
        
        self.time = time.astype('datetime64[ns]').astype(np.float64)
        self.u10 = u10
        self.v10 = v10
        self.nodes = nodes
        super().__init__(
            number_of_neighbor_layers=number_of_neighbor_layers,
            vship=vship,
            WD_min=WD_min,
            WVPI=WVPI,
            WWL=WWL,
            LWL=LWL,
            ukc=ukc,
            nl=nl,
            blend=blend,
            nodes_index=nodes_index,
            *args,
            **kwargs
        )

    def load(self):
        
        WD = np.full(self.u10.shape, 50)  

        return {
            "time": self.time,
            "nodes": self.nodes,
            "u": self.u10,
            "v": self.v10,
            "water_depth": WD,
        }
    def parse(self):
        print("Loading hydrodynamic data...")
        self.load_hydrodynamic()
        print("Hydrodynamic data loaded.")

        
        if self.nodes_index is None:
            print("Reducing nodes...")
            self.nodes_index, self.LS = self.get_nodes()
            print("Node reduction completed.")
        else:
            self.nodes_index = self.nodes_index
            self.LS = None
            print("Using precomputed nodes index.")
    
        nodes = self.nodes[self.nodes_index]
        u = np.asarray(np.transpose(self.u))[self.nodes_index]
        v = np.asarray(np.transpose(self.v))[self.nodes_index]
        WD = np.asarray(np.transpose(self.WD))[self.nodes_index]
    
        print("Nodes, u, v, WD loaded for land.")
        self.nodes, self.u, self.v, self.WD = self.nodes_on_land(nodes, u, v, WD)
        self.tria = scipy.spatial.Delaunay(self.nodes)
        self.mask = np.full(self.u.shape, False)
        self.mask[self.WD < self.WD_min.max() + self.ukc] = True
        print("Mask applied.")
    
        
        print("Calculating edges...")
        graph0 = Graph()
        for from_node in range(len(self.nodes)):
            if from_node % 100 == 0:  # Print progress every 100 nodes
                print(f"Processed {from_node} nodes out of {len(self.nodes)}")
            to_nodes = functions.find_neighbors2(
                from_node, self.tria, self.number_of_neighbor_layers
            )
            for to_node in to_nodes:
                L = functions.haversine(self.nodes[from_node], self.nodes[int(to_node)])
                graph0.add_edge(from_node, int(to_node), L)
        self.graph = Graph()
        vship1 = self.vship[0]
        for edge in graph0.weights:
            for i in range(len(vship1)):
                for j in range(len(vship1)):
                    from_node = edge[0]
                    to_node = edge[1]
                    self.graph.add_edge((from_node, i), (to_node, j), 1)

        print("Edge calculation completed.")
        
        calc_weights = self.calc_weights_time
        self.weight_space = []
        self.weight_time = []
        self.weight_cost = []
        self.weight_co2 = []

        for vv in range(len(self.vship)):
            print(f"doing{vv}")
            graph_time = Graph()
            graph_space = Graph()
            graph_cost = Graph()
            graph_co2 = Graph()
            vship = self.vship[vv]
            WD_min = self.WD_min[vv]
            WVPI = self.WVPI[vv]
            for edge in graph0.weights:
                for i in range(len(vship)):
                    for j in range(len(vship)):
                        from_node = edge[0]
                        to_node = edge[1]

                        L, W, euros, co2 = calc_weights(
                            edge,
                            i,
                            j,
                            vship,
                            WD_min,
                            WVPI,
                            self,
                            self.compute_cost,
                            self.compute_co2,
                            self.number_of_neighbor_layers,
                        )

                        graph_time.add_edge((from_node, i), (to_node, j), W)
                        graph_space.add_edge((from_node, i), (to_node, j), L)
                        graph_cost.add_edge((from_node, i), (to_node, j), euros)
                        graph_co2.add_edge((from_node, i), (to_node, j), co2)

            if "space" in self.optimization_type:
                self.weight_space.append(graph_space)
            if "time" in self.optimization_type:
                self.weight_time.append(graph_time)
            if "cost" in self.optimization_type:
                self.weight_cost.append(graph_cost)
            if "co2" in self.optimization_type:
                self.weight_co2.append(graph_co2)

    def calc_weights_time(
        self,
        edge,
        i,
        j,
        vship,
        WD_min,
        WVPI,
        self_f,
        compute_cost,
        compute_co2,
        number_of_neighbor_layers,
    ):
        
        from_node = edge[0]
        W = (
            functions.costfunction_timeseries(
                edge,
                vship[j],
                WD_min,
                self_f,
                WVPI,
                number_of_neighbor_layers,
                self_f.tria,
            )
            + self_f.t
        )
        W = self.fifo_maker(W, self_f.mask[from_node]) - self_f.t

        L = functions.costfunction_spaceseries(
            edge, vship[j], WD_min, self_f, WVPI, number_of_neighbor_layers, self_f.tria
        )
        L = L + np.arange(len(L)) * (1 / len(L))
        L = self.fifo_maker(L, self_f.mask[from_node]) - np.arange(len(L)) * (
            1 / len(L)
        )
        euros = compute_cost(W, vship[j])
        co2 = compute_co2(W, vship[j])

        return L, W, euros, co2

    @staticmethod
    def fifo_maker(y, N1):
        
        arg = np.squeeze(argrelextrema(y, np.less))
        if arg.shape == ():
            arg = np.array([arg])
        else:
            None
        y_FIFO = 1 * y
        for a in arg:
            loc = np.argwhere(y[: a + 1] <= y[a])[-2:]
            if loc.shape == (2, 1):
                if (N1[int(loc[0]) : int(loc[1])]).any():

                    None
                else:
                    y_FIFO[int(loc[0]) : int(loc[1])] = y[a].item()

        return y_FIFO

        


# In[28]:


roadmap = CustomRoadmap(
    u10=u,  
    v10=v,  
    time=time,  
    nodes=nodes,
    dx_min=0.1 
)


class Graph:
    

    def __init__(self):
        
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):

        self.edges[from_node].append(to_node)
        self.weights[(from_node, to_node)] = weight



# In[29]:


from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import scipy.spatial
from numpy import ma
from scipy.signal import argrelextrema

import halem.functions as functions

roadmap.parse()

# In[32]:


t0 = "01/01/2024 00:00:00"

start = ( 64, 28)
stop = ( 51, 11)
v_max = 8


path_t, time_t, dist_t = HALEM_time(start, stop, t0, v_max, roadmap)
path_s, time_s, dist_s = HALEM_space(start, stop, t0, v_max, roadmap)
path_c, time_c, dist_c = HALEM_cost(start, stop, t0, v_max, roadmap)
path_p, time_p, dist_p = HALEM_co2(start, stop, t0, v_max, roadmap)


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
time_df = pd.DataFrame(path_t)
longitudes_t = [time_df.loc[i,0] for i in range(time_df.shape[0])]  
latitudes_t = [time_df.loc[i,1] for i in range(time_df.shape[0])]

plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())


ax.set_extent([40, 80, 0, 30], crs=ccrs.PlateCarree())


ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
flag = True


for lat, lon in zip(latitudes_t, longitudes_t):
    if flag:
        plt.plot(lon, lat, marker='o', color='red', markersize=3, transform=ccrs.PlateCarree(), label="Time Optimal")
        flag=False
    else:
        plt.plot(lon, lat, marker='o', color='red', markersize=3, transform=ccrs.PlateCarree())
        
flag=True

time_df = pd.DataFrame(path_s)
longitudes_co = [time_df.loc[i,0] for i in range(time_df.shape[0])]
latitudes_co = [time_df.loc[i,1] for i in range(time_df.shape[0])]
for lat, lon in zip(latitudes_co, longitudes_co):
    if flag:
        plt.plot(lon, lat, marker='o', color='yellow', markersize=3, transform=ccrs.PlateCarree(), label="Emmision Optimial")
        flag = False
    else:
        plt.plot(lon, lat, marker='o', color='yellow', markersize=3, transform=ccrs.PlateCarree())
time_df = pd.DataFrame(path_c)
longitudes_c = [time_df.loc[i,0] for i in range(time_df.shape[0])]
latitudes_c = [time_df.loc[i,1] for i in range(time_df.shape[0])]
flag = True
for lat, lon in zip(latitudes_c, longitudes_c):
    if flag:
        plt.plot(lon, lat, marker='o', color='blue', markersize=3, transform=ccrs.PlateCarree(), label="Cost Optimal")
        flag= False
    else:
        plt.plot(lon, lat, marker='o', color='blue', markersize=3, transform=ccrs.PlateCarree())

flag=True
Lon, Lat = np.meshgrid(longitudes, latitudes) 
for i in range(u.shape[0]):
    if flag:
        plt.quiver(Lon, Lat, u[i], v[i], scale=500, color='cyan', alpha=0.01, transform=ccrs.PlateCarree(), label="Wind Vectors")
        flag=False
    else:
        plt.quiver(Lon, Lat, u[i], v[i], scale=500, color='cyan', alpha=0.01, transform=ccrs.PlateCarree())


ax.gridlines(draw_labels=True)


plt.title('Optimized Path with Wind Vectors')
plt.legend()

plt.show()








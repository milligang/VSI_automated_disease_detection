from my_bottleneck import *
from my_read_graph import read_graph
import numpy as np

# Example NumPy array
dataset_1 = '../Data/Dataset_1/NEFI_graphs_VK/'
txt_files, _  = dataset_1_files(dataset_1)

X_1 = np.load('Dataset_1_output/set1_PCA.npy')

def gd(txt_files):
    x_vals = []
    for file in txt_files:
        gd = geodesic_distance(read_graph(file))
        avg = sum(gd.values())/len(gd)
        x_vals.append(avg)
    return x_vals
def edges(txt_files):
    x_vals = []
    for file in txt_files:
        G = read_graph(file)
        x_vals.append(G.number_of_edges())
    return x_vals

x_vals = edges(txt_files)
fig, ax = plt.subplots()
ax.plot(x_vals, ((X_1[:,0])*(X_1[:,0]) + (X_1[:,1])*(X_1[:,1]))**0.5, 'o')

ax.set(xlabel='Edges', ylabel='PCA Radius')

plt.show()
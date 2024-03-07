import networkx as nx
import matplotlib.pyplot as plt
import os
from ripser import ripser
import numpy as np
import gudhi as gd
from itertools import combinations
from operator import itemgetter
from persim import plot_diagrams
from gudhi.persistence_graphical_tools import plot_persistence_diagram
from my_read_graph import *
from sklearn import datasets
import matplotlib
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, LogLocator)

# path to the data from the graph to be read in
graph_path = '../Data/Dataset_1/NEFI_graphs_VK/webs_im0077_#0_11h10m58s/Graph Filtering_smooth_2_nodes_im0077.txt'
nxgraph = read_graph(graph_path)

# remove # to plot graph
# plot_graph(nxgraph)

# calculate the central node
def my_central_node_ID(graph_in):
    central = nx.betweenness_centrality(graph_in,weight="weight")
    #determine which node is the central node
    cent_max = 0
    
    #determine max centrality
    for x in central:
        if central[x] > cent_max:
            cent_max = central[x]

    #extract which nodes have max centrality
    central_node_list = []
    for x in central:
        if central[x] == cent_max:
            central_node_list.append(x)

    #In case of multiple most central nodes -- we pick left-most
    central_node = central_node_list[0]
    for x in central_node_list:
        if x < central_node:
            central_node = x

    return central_node

central_node = my_central_node_ID(nxgraph) # this output matches the central node in geodesic2.py code (which uses coordinates)

# calculate geodesic distances for nxgraph
geodesic_distance = nx.single_source_dijkstra_path_length(nxgraph, central_node, cutoff=None, weight='weight')

# create persistence diagram
def find_connections(n, alive_dict):
    connected_nodes = []
    for (head, tail) in nxgraph.edges:
        if head == n:
            connected_nodes.append(tail)
        elif tail == n:
            connected_nodes.append(head)
    alive_connections = {}
    for node in connected_nodes:
        if node in alive_dict:
            alive_connections[node] = alive_dict[node]
    return alive_connections    
 
def compare(connections_dict):
    temp_tuple = (0, (0, 0))
    # find oldest node in connect_dict
    for n, (b, d) in connections_dict.items():
        if d > temp_tuple[1][1]:
            temp_tuple = (n, (b, d))
    oldest_node, (b_o, d_o) = temp_tuple
    return oldest_node, (b_o, d_o + 1)

def transfer(alive_dict, pers_dict):
    # transfer everything in alive_dict from pers_dict[0] to pers_dict[1]
    for key, _ in alive_dict.items():
        if key in pers_dict[0]:
            pers_dict[1][key] = pers_dict[0].pop(key)
    return pers_dict

def persistence(gd_dict):
    pers_dict = [{}, {}]
    while len(gd_dict) > 1:
        temp_list = []
        *_, max_gd = gd_dict.values()
        for n, gd in gd_dict.items():
            alive = pers_dict[0]
            if gd == max_gd:
                connections = find_connections(n, alive)
                connections = {key:(b, d) for key, (b, d) in connections.items() if b != gd}
                if len(connections) == 0:
                    alive[n] = (gd, 1)
                else:
                    key, value = compare(connections)
                    del alive[key]
                    alive[n] = value
                    del connections[key]
                    pers_dict = transfer(connections, pers_dict)
                temp_list.append(n)
        for i in temp_list:
            if i in gd_dict:
                del gd_dict[i]
    return pers_dict
# error: has 143: 34, 36
# works for simple test graph

# graph persistence diagram

def graph_pers(pers_dict):
    data = np.array(list(pers_dict[1].values()))
    dgms = ripser(data)['dgms']
    plot_diagrams(dgms, show=True)

graph_pers(persistence(geodesic_distance))

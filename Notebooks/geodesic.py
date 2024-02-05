
import glob
import networkx as nx
import numpy as np

from my_custom_functions import network_descriptors, geodesic_distancematrix

# The following code is for finding the edges, nodes, and edge lengths. Comes from Fractal dimension.ipynb.

dataset = "STARE1"
nefi_output_folder = "../Data/Dataset_1/NEFI_graphs/*/"

file_name = "im"
nefi_outputs = glob.glob(f"{nefi_output_folder}*.txt")

nodes = []
edges = []
edge_lengths = []

nums = np.array([1,2,3,4,5,44,77,81,82,139,162, 163, 235, 236, 239, 240, 255, 291, 319, 324])

for num in nums:
    
    num_str = f"{str(num).zfill(4)}"
    
    #find nefi output file
    nefi_output = [s for s in nefi_outputs if num_str in s]
    #ensure there is only one location in this list
    assert len(nefi_output)==1
    #read in graph
    graph_in = nx.read_multiline_adjlist(nefi_output[0],delimiter='|')
    edges_tmp, nodes_tmp, edge_lengths_tmp = network_descriptors(graph_in)
    edges.append(edges_tmp)
    nodes.append(nodes_tmp)
    edge_lengths.append(edge_lengths_tmp)


import glob
import networkx as nx
import numpy as np

from my_custom_functions import network_descriptors, geodesic_distancematrix

###########################################################
# The following code is for finding the edges, nodes, and edge lengths. Comes from Fractal dimension.ipynb.

dataset = "HRF"
nefi_output_folder = "../Data/HRF_Dataset_1/NEFI_graphs/*/"
image_folder = "../Data/HRF_Dataset_1/Provided_masks/"
write_folder = "../Results/HRF_Dataset_1/"

file_name = "im"
nefi_outputs = glob.glob(f"{nefi_output_folder}*.txt")

nodes = []
edges = []
edge_lengths = []

nums = np.arange(1,46)
mat = np.load("../Data/Diagnoses/image_diagnoses_HRF.npy",allow_pickle=True).item()

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

print (edges, nodes, edge_lengths)

###########################################################

geodesic_distancematrix(nodes, edges, edge_lengths)


import glob
import networkx as nx
import numpy as np

from custom_functions import *
from my_custom_functions import geodesic_distancematrix

'''
Current Progress:
1) central node issue: given a list of nodes, which is the central node?
2) what is graph_in for the weighted graph function?
'''

# network x graph for edge length calculations
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
    xgraph_in = nx.read_multiline_adjlist(nefi_output[0],delimiter='|')
    edges_tmp, nodes_tmp, edge_lengths_tmp = network_descriptors(xgraph_in)
    edges.append(edges_tmp)
    nodes.append(nodes_tmp)
    # edge_lengths is a list of float lists, one float list per graph
    edge_lengths.append(edge_lengths_tmp)

graph_path = '../Data/Dataset_1/NEFI_graphs/webs_im0077_#0_12h03m01s/Graph Filtering_smooth_2_nodes_im0077.png'

# function for nodes
weighted_graph = Graph_to_weighted(graph_in)
nodesList = get_nodes(weighted_graph)

"central node is the coords of the central node, but we want the central node's number in the list of nodes"
central_node = central_node_ID(weighted_graph)

# call geodesic function
# geodesic_distancematrix(nodesList, edges, edge_lengths, central_node)
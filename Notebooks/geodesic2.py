
import glob
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from custom_functions import *
from my_custom_functions import *

'''
Tasks:
graph_in works for a single graph- now want to run it for all the graphs in the list
get geodesic distance function to work --> need to check/fix: edges, edge_lengths, central_node
for num in nums is not returning what I am looking for
'''

# network x graph for edge length calculations
dataset = "STARE1"
nefi_output_folder = "../Data/Dataset_1/NEFI_graphs_VK/*/"

file_name = "im"
nefi_outputs = glob.glob(f"{nefi_output_folder}*.txt")

nodes_count_all = []
edges_count_all = []
edge_lengths = []
 
# currently only working with graph 77
#nums = np.array([1,2,3,4,5,44,77,81,82,139,162, 163, 235, 236, 239, 240, 255, 291, 319, 324])
nums = np.array([77])

# updates list of number of edges, number of nodes, and average edge length given list of graphs (nums)
for num in nums:
    num_str = f"{str(num).zfill(4)}" # returns a 4 digit integer, the ending digits are num and the remaining front digits are 0
    #find nefi output file
    nefi_output = [s for s in nefi_outputs if num_str in s]
    #ensure there is only one location in this list
    assert len(nefi_output)==1
    #read in graph
    xgraph_in = nx.read_multiline_adjlist(nefi_output[0],delimiter='|')
    edges_count_single, nodes_count_single, edge_lengths_single = my_network_descriptors(xgraph_in)
    edges_count_all.append(edges_count_single) # number of nodes for all the graphs
    nodes_count_all.append(nodes_count_single) # number of edges for all the graphs
    edge_lengths.append(edge_lengths_single) # list of (float list of the average length for each edge) for all the graphs

# path to the unweighted graph I want to use as graph_in
graph_path = '../Data/Dataset_1/NEFI_graphs_VK/webs_im0077_#0_11h10m58s/Graph Filtering_smooth_2_nodes_im0077.txt'
# read in graph from path
graph_in = read_graph_data(graph_path)

weighted_graph = Graph_to_weighted(graph_in)
node_list = get_nodes(weighted_graph)
edge_list = graph_in.edges
# edit nodeList so it is compatible with geodesic functions
node_list_tuples = [tuple(node) for node in node_list]
# edit edges_list:
# parse string representations of node coordinates into tuples of integers
parse_coordinate = lambda coord_str: tuple(map(int, coord_str.strip('()').split(', ')))
#convert string representations to tuples of integers using list comprehension
edge_list_ints = [(parse_coordinate(start), parse_coordinate(end)) for start, end in edge_list]

# central node returns the coordinates of the central node 
central_node = central_node_ID(weighted_graph)

# call geodesic function
shortest_path = geodesic_distancematrix(node_list_tuples, edge_list_ints, edge_lengths[0], central_node)

# make networkx graph
G = nx.Graph()
G.add_nodes_from(node_list_tuples)
for edgeID,edge in enumerate(edge_list_ints):
    G.add_edge(edge_list_ints[0],edge_list_ints[1],weight=edge_lengths[0][edgeID])

dict = {i: array for i, array in enumerate(node_list)}
# create plot
# print(np.min(np.array(edge_list_ints)))
print(node_list_tuples)
nx.draw(G, dict)
plt.show()
plt.draw(G)
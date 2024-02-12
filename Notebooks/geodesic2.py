
import glob
import networkx as nx
import numpy as np

from custom_functions import *
from my_custom_functions import geodesic_distancematrix

'''
Tasks:
graph_in works for a single graph- now want to run it for all the graphs in the list
get geodesic distance function to work --> need to check/fix: edges, edge_lengths, central_node
for num in nums is not returning what I am looking for
'''

# network x graph for edge length calculations
dataset = "STARE1"
nefi_output_folder = "../Data/Dataset_1/NEFI_graphs/*/"

file_name = "im"
nefi_outputs = glob.glob(f"{nefi_output_folder}*.txt")

nodes = []
edges = []
edge_lengths = []
 
# currently only working with graph 77
#nums = np.array([1,2,3,4,5,44,77,81,82,139,162, 163, 235, 236, 239, 240, 255, 291, 319, 324])
nums = np.array([77])

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

# path to the unweighted graph I want to use as graph_in
graph_path = '../Data/Dataset_1/NEFI_graphs_VK/webs_im0077_#0_11h10m58s/Graph Filtering_smooth_2_nodes_im0077.txt'

# read data, returning edges and nodes
def read_graph_data(file_path):
    graph_in = nx.Graph()

    with open(file_path, 'r') as file:
        for line in file:
            # Skip comment lines
            if line.startswith('#'):
                continue
            
            # Split the line into coordinates and attributes
            parts = line.strip().split('|')
            head_coordinates = tuple(map(int, parts[0].strip('()').split(',')))
            attributes = eval(parts[1])  # Safely evaluate the attributes part

            # Add the head node to the graph
            head_node = str(head_coordinates)
            graph_in.add_node(head_node)

            # If the head node has edges, add them to the graph
            if isinstance(attributes, int) and attributes > 0:
                for _ in range(attributes):
                    # Split the line to get the coordinates of the tail node
                    tail_coordinates = tuple(map(int, file.readline().strip().split('|')[0].strip('()').split(',')))
                    tail_node = str(tail_coordinates)
                    graph_in.add_node(tail_node)  # Add the tail node to the graph
                    graph_in.add_edge(head_node, tail_node)  # Add the edge between head and tail nodes

    return graph_in
graph_in = read_graph_data(graph_path)

# function for nodes
weighted_graph = Graph_to_weighted(graph_in)
nodeList = get_nodes(weighted_graph)

# edit nodeList so it is compatible with geodesic functions
nodeList_tuples = [tuple(node) for node in nodeList]

# central node returns the coordinates of the central node 
central_node = central_node_ID(weighted_graph)

# call geodesic function
geodesic_distancematrix(nodeList_tuples, edges, edge_lengths, central_node)
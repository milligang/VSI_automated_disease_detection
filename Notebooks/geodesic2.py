
import glob
import networkx as nx
import numpy as np

from custom_functions import *
from my_custom_functions import geodesic_distancematrix

'''
Current Progress:
1) central node issue: given a list of nodes, which is the central node?
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

graph_path = '../Data/Dataset_1/NEFI_graphs_VK/webs_im0077_#0_11h10m58s/Graph Filtering_smooth_2_nodes_im0077.txt'
def process_graph_from_file(file_path):
    """
    Read data from a file, construct a NetworkX graph, and populate it with nodes and edges with attributes.
    """
    graph_in = nx.Graph()

    try:
        with open(file_path, 'r') as file:
            node_buffer = None  # Buffer to store the current node while processing edges
            for line in file:
                line = line.strip()
                if line.startswith('#'):
                    continue
                parts = line.split('|')
                coordinates = tuple(map(int, parts[0].strip('()').split(',')))
                attributes_str = parts[1].strip()
                if attributes_str:
                    # Parse attributes to extract edge information
                    attributes = eval(attributes_str)
                    for node, edge_attr in attributes.items():
                        if node != 'pixels':  # Exclude 'pixels' from edge attributes
                            node_coord = tuple(map(int, node.strip('()').split(',')))
                            graph_in.add_edge(coordinates, node_coord, **edge_attr)
                else:
                    # If the line contains an integer, it represents a node with outgoing edges
                    num_outgoing_edges = int(line)
                    node_buffer = coordinates  # Store the current node
                    # Process outgoing edges for this node based on the integer value
                    for _ in range(num_outgoing_edges):
                        # Read the next line to get edge information
                        next_line = next(file).strip()
                        if next_line.startswith('{'):
                            # If the next line contains a dictionary, it represents an edge
                            edge_attributes = eval(next_line)
                            for node, edge_attr in edge_attributes.items():
                                if node != 'pixels':
                                    node_coord = tuple(map(int, node.strip('()').split(',')))
                                    graph_in.add_edge(node_buffer, node_coord, **edge_attr)
                        else:
                            # If the next line contains an integer, it represents another node
                            # We need to process it in the next iteration of the loop
                            node_buffer = tuple(map(int, next_line.strip('()').split(',')))
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

    return graph_in


graph_in = process_graph_from_file(graph_path)

# function for nodes
weighted_graph = Graph_to_weighted(graph_in)
nodesList = get_nodes(weighted_graph)

"central node is the coords of the central node, but we want the central node's number in the list of nodes"
#central_node = central_node_ID(weighted_graph)

# call geodesic function
#geodesic_distancematrix(nodesList, edges, edge_lengths, central_node)
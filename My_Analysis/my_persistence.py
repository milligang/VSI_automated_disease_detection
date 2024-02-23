import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

# read in data as a networkx graph
def read_graph(graph_path):
    # Create an empty NetworkX graph
    G = nx.Graph()
    # Track unique node labels
    node_labels = {}
    try: 
        with open(graph_path, 'r') as file:
            for line in file:
                line = line.strip()
                # Skip comment lines
                if line.startswith('#') or not line:
                    continue
                
                # Split the line into coordinates and attributes
                parts = line.split('|')
                head_coordinates = tuple(map(int, parts[0].strip('()').split(',')))
                attributes = eval(parts[1])  # Safely evaluate the attributes part

                # If the head node has edges, add them to the graph
                if isinstance(attributes, int) and attributes > 0:
                    head_node = str(head_coordinates)  # Convert the head coordinates to a string

                    # Add the head node to the graph with a unique label
                    if head_node not in node_labels:
                        node_labels[head_node] = len(node_labels)
                        G.add_node(node_labels[head_node])

                    for _ in range(attributes):
                        # Split the line to get the coordinates of the tail node
                        tail_line = file.readline().strip()
                        tail_coordinates = tuple(map(int, tail_line.split('|')[0].strip('()').split(',')))
                        tail_node = str(tail_coordinates)  # Convert the tail coordinates to a string

                        # Add the tail node to the graph with a unique label
                        if tail_node not in node_labels:
                            node_labels[tail_node] = len(node_labels)
                            G.add_node(node_labels[tail_node])

                        # Add the edge between head and tail nodes
                        G.add_edge(node_labels[head_node], node_labels[tail_node])
    except FileNotFoundError:
        print(f"File not found: {graph_path}")
    except Exception as e:
        print(f"Error processing file: {e}")
    return G

# path to the data from the graph to be read in
graph_path = '../Data/Dataset_1/NEFI_graphs_VK/webs_im0077_#0_11h10m58s/Graph Filtering_smooth_2_nodes_im0077.txt'

nxgraph = read_graph(graph_path)

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

# create functions to make persistence diagram
# create the initial persistence diagram
x_coords, y_coords = [], []
plt.scatter(x_coords, y_coords, color='blue', label='Initial Points')

# connected_nodes takes in a node and returns a list of all of the nodes connected to it (moving up the branch)
def connected_nodes(n):
    tail_nodes = []
    for (head, tail) in nxgraph.edges:
        if head == n:
            tail_nodes.append(tail)
    return tail_nodes

# graph takes a list of birth, death pairs as input and adds these to the persistance diagram
def graph(tuple_list):
    xy = zip(*tuple_list)
    x_coords.append(xy[0])
    y_coords.append(xy[1])
    plt.scatter(x_coords, y_coords, color='red', label='New Points')
    plt.legend()

# compare takes in a list of birth, death tuples for a single branch and returns the tuple with highest death value, adding the rest of the tuples to the persistence diagram
def compare(tuple_list): 
    # find the tuple with the highest death value, the 'most persistent'
    max_tuple = max(tuple_list, key = itemgetter(1)) # if multiple max_tuples, returns the first one
    tuple_list.remove(max_tuple)
    max_tuple = max_tuple[0], max_tuple[1] + 1
    graph(tuple_list)
    return max_tuple

# branches takes a node n as input and recursivley calls itself to return a birth, death tuple associated with the branch 
def branches(n):
    connections = connected_nodes(n)
    # create a list to store the birth, death tuple of each node connected to n, including n itself
    persistence_values = [(geodesic_distance[n], 0)]
    # for each node connected to n, add branches(connection) to the list
    for connection in connections:
            persistence_values.append(branches(connection))
    return compare(persistence_values)

branches(central_node)
plt.show()
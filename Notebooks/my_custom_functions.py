import numpy as np
import networkx as nx
import matplotlib

def my_network_descriptors(graph_in):
    
    '''
    extract edges, nodes, and average edge length (pixels) from graph
    
    inputs
    
    graph_in:      network x graph
    
    outputs
    
    edges:         number of edges
    nodes:         number of nodes
    edge_lengths:  length of each edge
    
    '''
    
    edges = graph_in.number_of_edges()
    nodes = graph_in.number_of_nodes()
    
    edge_length_tmp = []
    for n in list(graph_in.edges()):

        #extract two nodes from node names
        node1_str = n[0]
        node1 = tuple(map(int, node1_str[1:-1].split(', ')))
        node2_str = n[1]
        node2 = tuple(map(int, node2_str[1:-1].split(', ')))

        #determine weight via Euclidean distance
        edge_length_tmp.append(np.linalg.norm((node1[0] - node2[0],node1[1] - node2[1])))
    
    return edges, nodes, edge_length_tmp

def geodesic_distancematrix(nodeList, edgeList, edgeLength, centralNode):
	""" 
	Construct a geodesic distance matrix using dijstra shortest paths 
	
	inputs
	
	nodeList: list of tuples of the nodes
	edgeList: the edges 
	edgeLength: list of length of each edge
	centralNode: coordinates of the central node
	"""
	Nnodes = len(nodeList) # the number of nodes
	# Create the graph structure required for networkx
	G = nx.Graph()
	G.add_nodes_from(nodeList)
	for edgeID,edge in enumerate(edgeList):
		G.add_edge(edgeList[0],edgeList[1],weight=edgeLength[edgeID])
	# Find the shortest paths between nodes and source
	shortest_path = nx.single_source_dijkstra_path_length(G, centralNode, cutoff = None, weight='weight')

	return shortest_path

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
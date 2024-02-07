import numpy as np
import networkx as nx

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
	""" Construct a geodesic distance matrix using dijstra shortest paths """
	
	Nnodes = len(nodeList)
	
	# Create the graph structure required for networkx
	G = nx.Graph()
	G.add_nodes_from(nodeList)

	for edgeID,edge in enumerate(edgeList):
		G.add_edge(edgeList[0],edgeList[1],weight=edgeLength[edgeID])

	# Find the shortest paths between nodes and source, and unpack in distance matrix
	DistanceDict = dict(nx.single_source_dijkstra_path_length(G, centralNode))
	DM = np.ones((Nnodes,Nnodes))*-1
	for i in range(Nnodes):
		for j in range(Nnodes):
			DM[i,j] = DistanceDict[i][j]
			
	return DM


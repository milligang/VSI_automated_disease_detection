import networkx as nx
import matplotlib.pyplot as plt

# ---------- read in data as networkx graph ----------
# function that reads in data as a networkx graph
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

pos = nx.spring_layout(nxgraph)  # Define positions of nodes
nx.draw(nxgraph, pos, with_labels=False, node_size=20, node_color='skyblue')  # Draw the graph
plt.title('Graph Visualization')  # Set the title of the plot
plt.show()  # Show the plot

# ----- calculate the geodesic distance to each node -----

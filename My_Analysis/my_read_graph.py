import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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
                        G.add_node(node_labels[head_node], pos=(head_coordinates[1],-head_coordinates[0]))

                    for _ in range(attributes):
                        # Split the line to get the coordinates of the tail node
                        tail_line = file.readline().strip()
                        tail_coordinates = tuple(map(int, tail_line.split('|')[0].strip('()').split(',')))
                        tail_node = str(tail_coordinates)  # Convert the tail coordinates to a string

                        # Add the tail node to the graph with a unique label
                        if tail_node not in node_labels:
                            node_labels[tail_node] = len(node_labels)
                            G.add_node(node_labels[tail_node], pos=(tail_coordinates[1],-tail_coordinates[0]))

                        # Add the edge between head and tail nodes
                        edge_length = float((tail_line.split('|')[1].split(':')[2].split(',')[0]))
                        G.add_edge(node_labels[head_node], node_labels[tail_node], length = edge_length)
    except FileNotFoundError:
        print(f"File not found: {graph_path}")
    except Exception as e:
        print(f"Error processing file: {e}")
    return G


def obtain_diagnoses(data_set, results_dir, data_type="PI"):
    '''
    obtain the disease classifications of each VSI
    
    inputs:
    
    data_type : "PI" for persistence images
    
    outputs:
    
        ID : IDs of each VSI
        data : dictionary containing TDA descriptor vectors for each filtration for each VSI
        diag : disease classification for each VSI
    '''
    if "stare" == data_set:
        num_images = 402
        diagnosis_keys = np.load("../Data/Diagnoses/image_diagnoses.npy",allow_pickle=True,encoding="latin1").item()
    elif data_set == "HRF":
        num_images = 45
        diagnosis_keys = np.load("../Data/Diagnoses/image_diagnoses_HRF.npy",allow_pickle=True,encoding="latin1").item()
    elif "all" == data_set:
        num_images = 161
        diagnosis_keys = np.load("../Data/Diagnoses/image_diagnoses_all.npy",allow_pickle=True,encoding="latin1").item()
            
    ID = np.zeros((num_images,),dtype=int)

    filtration_type = ['pers']
        
    if data_type == "PI":
        data_format = np.zeros((num_images,2*2500))

        data = {}
        for key in filtration_type:
            data[key] = np.copy(data_format)

    nums = list(diagnosis_keys['image_diagnoses'])
    diag = np.zeros((num_images,4))
    diag[:] = np.nan
    no_data_exists = []
    for i,num in enumerate(nums):

        ID[i] = i+1
        diagnosis = diagnosis_keys['image_diagnoses'][num]

        for j,d in enumerate(diagnosis):
            diag[i,j] = d

        try:    
            for key in filtration_type:                      
                file_name = (results_dir + "DS1_im" + num + "_" + key + "_PIR.npy")
                mat = np.load(file_name,encoding="latin1",allow_pickle=True).item()
                data[key][i,:] = np.hstack([mat['Ip'][0].reshape(-1),mat['Ip'][1].reshape(-1)])                     
        except:
            no_data_exists.append(i)

    ID = np.delete(ID,no_data_exists,axis=0)
    for key in filtration_type:
        data[key] = np.delete(data[key],no_data_exists,axis=0)
    diag = np.delete(diag,no_data_exists,axis=0)

    return ID, data, diag

def plot_graph(nxgraph):
    pos = nx.get_node_attributes(nxgraph, 'pos')
    return nx.draw(nxgraph, pos, with_labels = False, node_size = 10)
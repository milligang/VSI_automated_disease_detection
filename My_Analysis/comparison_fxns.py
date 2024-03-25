from my_bottleneck import *

dataset_1 = ['../Data/Dataset_1/NEFI_graphs_VK/']

def node_compare(dataset):
    txt_files, files_order = dataset_1_files(dataset)
    x_values = []
    y_values = bottle(dataset, False)[0]
    for txt_file in txt_files:
        G = read_graph(txt_file)
        x_values.appen(G.number_of_Nodes)
    print(x_values, y_values)
node_compare(dataset_1)
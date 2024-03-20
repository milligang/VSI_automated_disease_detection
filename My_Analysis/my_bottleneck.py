# compute the bottleneck distance
from my_persistence import *
from my_read_graph import read_graph
from os import walk
from persim import bottleneck

dataset = '../Data/Dataset_1/NEFI_graphs_VK/'
file_nums = [1, 2, 3, 4, 5, 44, 77, 81, 82, 139, 162, 163, 235, 236, 240, 255, 291, 319, 324]
txt_files = []

#note: this doesn't include the files where the txt is the second in the dir, not the first
for (root, _, txt_file) in walk(dataset):
    for num in file_nums:
        num_str = f"{str(num).zfill(4)}"
        name = f"{'_im'}{num_str}{'.txt'}"
        if name in txt_file[0]:
            txt_files.append(f"{root}{'/'}{txt_file[0]}")

# returns list of persistence coordinates for each dataset
def dgms(txt_files):
    dgms_collection = {}
    for txt_file in txt_files:
        graph_in = read_graph(txt_file)
        dgms_collection[txt_file] = pers_coords(graph_in)
    return dgms_collection

# computes the bottleneck distance for list of datasets
def bottle(txt_files):
    dgms_collection = dgms(txt_files)
    num_files = len(txt_files)
    DM_dataset = np.zeros((num_files, num_files))
    for i, (data1, dgms1) in enumerate(dgms_collection.items()):
        for j, (data2, dgms2) in enumerate(dgms_collection.items()):
            if data1 != data2:
                dist = bottleneck(dgms1[1], dgms2[1])
                DM_dataset[i,j] = dist
                DM_dataset[j,i] = dist
bottle(txt_files)
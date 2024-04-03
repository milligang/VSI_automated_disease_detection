# compute the bottleneck distance
from my_persistence import *
from my_read_graph import *
import os
from persim import bottleneck
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, LogLocator, FormatStrFormatter, AutoMinorLocator)
from sklearn.decomposition import PCA

path1 = '../Data/Dataset_1/NEFI_graphs/'
path2 = '../Data/HRF_Dataset_1/NEFI_graphs/'
path3 = '../Data/Dataset_1/NEFI_graphs_VK/'

def dataset_1_files(dataset):
    txt_files = []
    files_order = []
    result_paths = []
    if os.path.normpath(dataset) == '../Data/Dataset_1/NEFI_graphs':  
        result_dir = "Dataset_1_output"
        file_nums = [1, 2, 3, 4, 5, 44, 77, 81, 82, 139, 162, 163, 235, 236, 239, 240, 255, 291, 319, 324]
    if os.path.normpath(dataset) == '../Data/Dataset_1/NEFI_graphs_VK':  
        result_dir = "Dataset_1_VK_output"
        file_nums = [1, 2, 3, 4, 5, 44, 77, 81, 82, 139, 162, 163, 235, 236, 239, 240, 255, 291, 319, 324]
    elif os.path.normpath(dataset) == '../Data/HRF_Dataset_1/NEFI_graphs':
        result_dir = "HRF_Dataset_1_output"
        file_nums = list(range(1, 46))
    else:
        raise ValueError('Invalid Dataset') 
    for (root, _, txt_file) in os.walk(dataset):
        for num in file_nums:
            num_str = f"{str(num).zfill(4)}"
            name = f"{'_im'}{num_str}{'.txt'}"
            if len(txt_file) > 1:
                if name in txt_file[0]:
                    files_order.append(num)
                    txt_files.append(f"{root}{'/'}{txt_file[0]}")
                    result_paths.append('My_Results/' + result_dir + '/DS1_im' + num_str + '_pers' + '_PIR.npy')
                elif name in txt_file[1]:
                    files_order.append(num)
                    txt_files.append(f"{root}{'/'}{txt_file[1]}")
                    result_paths.append('My_Results/' + result_dir + '/DS1_im' + num_str + '_pers' + '_PIR.npy')
    return txt_files, files_order, result_paths

# returns list of persistence coordinates for each dataset
def dgms(txt_files, result_paths):
    dgms_collection = {}
    for i in range(len(txt_files)):
        txt_file = txt_files[i]
        graph_in = read_graph(txt_file)
        dgms_collection[txt_file] = pers_coords(graph_in)
        np.save(result_paths[i], dgms_collection[txt_file])
    return dgms_collection

def find_pca(DM_dataset):
    pca = PCA(n_components=2)
    pca.fit(DM_dataset)
    X = pca.transform(DM_dataset)
    return X 

def find_dgms(txt_files, result_paths):
    len_results = len(result_paths)
    dgms_collection = {}
    if len_results > 0:
        for i in range(len_results):
            path = result_paths[i]
            # read in the file if it exists and is not empty
            if os.path.isfile(path):
                if os.path.getsize(path) > 0:
                    try:
                        data = np.load(path)
                        dgms_collection[txt_files[i]] = data
                    except Exception as e:
                        print(f"Error occurred while loading the file: {e}")
                        return None
    else:
        # there is no saved dgms for this data, so need to call dgms
        dgms_collection = dgms(txt_files, result_paths)
    return dgms_collection
            
                    
# computes the bottleneck distance and perform PCA for list of datasets
def bottle(dataset, graph): # graph is a bool, if true then show the graphs
    txt_files, files_order, result_paths = dataset_1_files(dataset)
    dgms_collection = find_dgms(txt_files, result_paths)
    num_files = len(txt_files)
    DM_dataset = np.zeros((num_files, num_files))
    for i, (data1, dgms1) in enumerate(dgms_collection.items()):
        for j, (data2, dgms2) in enumerate(dgms_collection.items()):
            if data1 != data2:
                dist = bottleneck(dgms1, dgms2)
                DM_dataset[i,j] = dist
                DM_dataset[j,i] = dist
    X = find_pca(DM_dataset)
    if graph:
        # graph bottleneck distance
        fig1,ax1 = plt.subplots()
        ax1.imshow(DM_dataset, interpolation='none')
        Nlabels = len(files_order)
        ax1.set_xticks(np.arange(Nlabels))
        ax1.set_yticks(np.arange(Nlabels))
        ax1.set_xticklabels(files_order)
        ax1.set_yticklabels(files_order)   
        ax1.tick_params(axis="x", labelrotation=90)

        # graph PCA
        fig2, ax2 = plt.subplots()
        ax2.scatter(X[:,0],X[:,1])
        for i, num in enumerate(files_order):
            ax2.annotate(num, (X[i,0],X[i,1]))
        ax2.xaxis.set_major_locator(MultipleLocator(20000))
        ax2.xaxis.set_minor_locator(MultipleLocator(4000))
        ax2.yaxis.set_major_locator(MultipleLocator(5000))
        ax2.yaxis.set_minor_locator(MultipleLocator(1000))
        ax2.set_xlabel(r" $\mathrm{PCA}_1$")
        ax2.set_ylabel(r"$\mathrm{PCA}_2$") 
        plt.show()
    else:
        return X
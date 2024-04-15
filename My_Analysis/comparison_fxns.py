import numpy as np

from my_bottleneck import *
from my_read_graph import *
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold

### Dataset_1 STARE Expert 1
'''data_set = "stare"
image_data = '../Data/Dataset_1/NEFI_graphs/'
image_dir = "../Data/Dataset_1/Provided_masks/"
results_dir = "My_Results/Dataset_1_output/"'''

### Dataset_1_VK STARE Expert 2
'''data_set = "stare2"
image_data = '../Data/Dataset_1/NEFI_graphs_VK'
image_dir = "../Data/Dataset_1/Provided_masks_VK/"
results_dir = "My_Results/Dataset_1_VK_output/"'''

#HRF

data_set = "HRF"
image_data = '../Data/HRF_Dataset_1/NEFI_graphs/'
image_dir = "../Data/HRF_Dataset_1/Provided_masks/"
retinal_image_folder = "../Data/HRF_Dataset_1/Provided_retinal_images/*.png"
results_dir = "My_Results/HRF_output/"


## all
'''data_set = "all"
image_dir = "../Data/all/Provided_masks/"
results_dir = "My_Results/all_output/"
'''

# returns list of certain values to graph against PCA values
# gd: average geodesic distance of each graph
def gd(txt_files):
    x_vals = []
    for file in txt_files:
        gd = geodesic_distance(read_graph(file))
        avg = sum(gd.values())/len(gd)
        x_vals.append(avg)
    return x_vals
#edges: number of edges of each graph
def edges(txt_files):
    x_vals = []
    for file in txt_files:
        G = read_graph(file)
        x_vals.append(G.number_of_edges())
    return x_vals

def graph_compare(dataset):
    txt_files, _  = dataset_files(dataset)
    X_1 = np.load('Dataset_1_output/set1_PCA.npy')
    # change x_vals for different comparisons
    x_vals = edges(txt_files)
    #create graph
    fig, ax = plt.subplots()
    ax.plot(x_vals, ((X_1[:,0])*(X_1[:,0]) + (X_1[:,1])*(X_1[:,1]))**0.5, 'o')
    ax.set(xlabel='Edges', ylabel='PCA Radius')
    plt.show()

def cross_val_prediction(X,y):
    
    '''
    Use support vector machines (SVMs) to predict y from X using 5-fold cross validation (CV)

    inputs

    X:    ndarray feature matrix of size (N,p), (N is # of features, p is dimension of each feature)
    y:    ndarray of classifications of size (N,1)

    outputs

    scores_mean: Averaged mean OOS score for each round of 5-fold CV
    scores_std:  Averaged standard deviation of each OOS for each round of 5-fold CV
    '''
    
    scores = []
    for j in np.arange(100):
        clf = svm.SVC(C=2.0,random_state=j)
        cv = KFold(n_splits = 5, shuffle=True, random_state = j)
        results = cross_val_score(clf, X, y, cv = cv, scoring = 'accuracy')
        scores.append(np.mean(results))

    scores = np.array(scores)    
    scores_mean = np.mean(scores)
    scores_std = np.std(scores)
    
    return scores_mean, scores_std

ID, data, diag = obtain_diagnoses(data_set, results_dir)
y = 1*(np.any(diag==0,axis=1))
X = data['bottle']               
mean, std = cross_val_prediction(X, y)
print(f"Mean: {100*np.round(mean,3)}, SD: {100*np.round(std,3)}")
# stare Mean: 35.2 %, SD: 8.0
# stare expert 2 Mean: 33.8%, SD: 9.4
# HRF Mean: 61.1%, SD: 3.4

# stare bottleneck Mean: 34.3%, SD: 8.7
# stare expert 2 bottleneck Mean: 44.0%, SD: 10.8
# HRF bottleneck Mean: 61.7%, SD: 2.9


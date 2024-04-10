import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
import matplotlib.image as mpimg
from sklearn.decomposition import PCA

from my_read_graph import *
from my_bottleneck import bottle

### Dataset_1 STARE Expert 1
'''
data_set = "stare"
image_data = '../Data/Dataset_1/NEFI_graphs/'
image_dir = "../Data/Dataset_1/Provided_masks/"
results_dir = "My_Results/Dataset_1_output/"
'''

### Dataset_1_VK STARE Expert 2
data_set = "stare2"
image_data = '../Data/Dataset_1/NEFI_graphs_VK'
image_dir = "../Data/Dataset_1/Provided_masks_VK/"
results_dir = "My_Results/Dataset_1_VK_output/"

#HRF
'''
data_set = "HRF"
image_data = '../Data/HRF_Dataset_1/NEFI_graphs/'
image_dir = "../Data/HRF_Dataset_1/Provided_masks/"
retinal_image_folder = "../Data/HRF_Dataset_1/Provided_retinal_images/*.png"
results_dir = "My_Results/HRF_output/"
'''

## all
'''data_set = "all"
image_dir = "../Data/all/Provided_masks/"
results_dir = "My_Results/all_output/"
'''

ID, data, diag = obtain_diagnoses(data_set, results_dir)

fontsize=24

y = 1*(np.any(diag==0,axis=1))

filtration = 'pers'

X_pca = data[filtration]

mpl.rc("xtick",labelsize=15)
mpl.rc("ytick",labelsize=15)
#mpl.rcParams['font.family'] = 'serif'
#mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']

    
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.scatter(X_pca[y==0,0],X_pca[y==0,1],c="r",label="Diseased")
ax.scatter(X_pca[y==1,0],X_pca[y==1,1],c="b",label="Normal")

#for j, txt in enumerate(ID):
#    ax.annotate(txt, (X_pca[j,0], X_pca[j,1]))

#plt.legend(loc=2,fontsize=fontsize)

ax.set_xlabel("PCA component 1",fontsize=fontsize)
ax.set_ylabel("PCA component 2",fontsize=fontsize)
ax.set_title("Persistence PCA space",fontsize=fontsize)

ax.set_xticks([])
ax.set_yticks([])
#ax.set_xlim([-2,0])
#ax.set_ylim([-2,0])
plt.show()

from matplotlib.ticker import MultipleLocator
ID, data, diag = obtain_diagnoses("stare", "My_Results/Dataset_1_output/")

# graph PCA
fig2, ax2 = plt.subplots()
ax2.scatter(data['pers'][:,0], data['pers'][:,1])
ax2.xaxis.set_major_locator(MultipleLocator(20000))
ax2.xaxis.set_minor_locator(MultipleLocator(4000))
ax2.set_xlabel(r" $\mathrm{PCA}_1$")
plt.show()
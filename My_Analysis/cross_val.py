import multiprocessing as mp
import numpy as np

from custom_fxns import *

### STARE Expert 1
'''
data_set = "stare"
image_dir = "../Data/Dataset_1/Provided_masks/"
file_name = "im"
data_name = "DS1_"
results_dir = "My_Results/Dataset_1_output/"
'''

'''### STARE Expert 2
data_set = "stare2"
image_dir = "../Data/Dataset_1/Provided_masks_VK/"
file_name = "im"
data_name = "DS1_"
results_dir = "My_Results/Dataset_1_VK_output/"'''

#HRF
data_set = "HRF"
image_dir = "../Data/HRF_Dataset_1/Provided_masks/"
retinal_image_folder = "../Data/HRF_Dataset_1/Provided_retinal_images/*.png"
file_name = "im"
data_name = "DS1_"
results_dir = "My_Results/HRF_output/"


if data_set == "HRF":
    nums = np.arange(1,46)
elif "stare" in data_set:
    nums = np.array([1,2,3,4,5,44,77,81,82,139,162, 163, 235, 236, 239, 240, 255, 291, 319, 324])

def pers_filtrations(num):
    num_str = f"{str(num).zfill(4)}"
	
    filename_header = (results_dir + data_name + file_name + num_str + "_" + "pers")
    diag = np.load(filename_header +"_PIR.npy")
    
    PI_o, PI_r = Persist_im(diag=diag, inf_val = 40,sigma = 1.0, filename_save = [filename_header+"_PIO2", filename_header+"_PIR2"])

stare = STARE_manager(image_dir = image_dir,
                     file_name = file_name, 
                     data_name = data_name,
                     results_dir=results_dir,
                     data_set=data_set)

ID, data, diag =  stare.obtain_diagnoses(data_type="PI")
filtrations = data.keys()

y = 1*(np.any(diag==0,axis=1))

features = ["b0","b1","b0 & b1"]

for filtration in filtrations:
    
    print(f"Results using {filtration} filtration:")
    
    X = stare.PCA_reduction(data[filtration],comp=2,data_type="PI")
    
    for i in np.arange(3):
        mean, std = cross_val_prediction(X[i],y)
        print(f"{features[i]}: {100*np.round(mean,3)}, SD: {100*np.round(std,3)}")
    
    print("")

    # stare.PCA_interpretation(filtration)

# stare.plot_labelings(diag)

'''
stare1: 
    b0: 25.7, SD: 9.0
    b1: 33.0, SD: 7.0
    b0 & b1: 25.7, SD: 9.0

stare2:
    b0: 35.2, SD: 7.7
    b1: 33.0, SD: 7.0
    b0 & b1: 35.0, SD: 7.7

HRF:
    b0: 64.0, SD: 1.9
    b1: 66.7, SD: 0.0
    b0 & b1: 64.0, SD: 1.9
'''
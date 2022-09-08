import pandas as pd
import numpy as np
import pdb, sys, colorsys, glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class STARE_manager:

	def __init__(self, image_dir, file_name, results_dir, data_name, data_set = "stare"):

		self.image_dir = image_dir
		self.file_name = file_name
		self.data_name = data_name
		self.results_dir = results_dir
		self.data_set = data_set        

		if "stare" in self.data_set.lower():
			mat = np.load("../Data/Diagnoses/image_diagnoses.npy",allow_pickle=True,encoding="latin1").item()
			self.diagnosis_keys = mat
		elif self.data_set == "HRF":
			mat = np.load("../Data/Diagnoses/image_diagnoses_HRF.npy",allow_pickle=True,encoding="latin1").item()
			self.diagnosis_keys = mat
		elif "all" in self.data_set.lower():
			mat = np.load("../Data/Diagnoses/image_diagnoses_all.npy",allow_pickle=True,encoding="latin1").item()
			self.diagnosis_keys = mat         
		else:
			raise Exception("No diagnose file given")            
		self.diagnoses_unique = list(self.diagnosis_keys['diagnosis_codes'].values())
        
	def obtain_diagnoses(self,data_type="Betti"):
		'''
		obtain the disease classifications of each VSI
		
		inputs:
		
		data_type : "Betti" for Betti curves, "PI" for persistence images
		
		outputs:
		
			ID : IDs of each VSI
			data : dictionary containing TDA descriptor vectors for each filtration for each VSI
			diag : disease classification for each VSI
		'''
		if "stare" in self.data_set.lower():
			num_images = 402
		elif self.data_set == "HRF":
			num_images = 45
		elif "all" in self.data_set.lower():
			num_images = 161
            
		ID = np.zeros((num_images,),dtype=int)

		filtration_type = ['inward','outward','flooding','VR']
        
		if data_type == "PI":
			data_format = np.zeros((num_images,2*2500))

			data = {}
			for key in filtration_type:
				data[key] = np.copy(data_format)

		elif data_type == "Betti":
			data_format = np.zeros((num_images,80))

			data = {}
			for key in filtration_type:
				data[key] = np.copy(data_format)

		nums = list(self.diagnosis_keys['image_diagnoses'])
		diag = np.zeros((num_images,4))
		diag[:] = np.nan
		no_data_exists = []
		for i,num in enumerate(nums):

			ID[i] = i+1
			diagnosis = self.diagnosis_keys['image_diagnoses'][num]

			for j,d in enumerate(diagnosis):
				diag[i,j] = d

			try:
				if data_type=="Betti":

					for key in filtration_type:
						file_name = (self.results_dir + self.data_name + self.file_name + 
						num + "_" + key +"_Betti.npy")
						mat_betti = np.load(file_name,encoding="latin1",allow_pickle=True).item()
						data[key][i,:] = np.hstack([mat_betti['b0'],mat_betti['b1']])
				else:# data_type=="PI":        
					for key in filtration_type:                      
						file_name = (self.results_dir + self.data_name + self.file_name + num + "_" + key +"_PIR.npy")
						mat = np.load(file_name,encoding="latin1",allow_pickle=True,).item()
						data[key][i,:] = np.hstack([mat['Ip'][0].reshape(-1),mat['Ip'][1].reshape(-1)])                       
			except:
				no_data_exists.append(i)

		ID = np.delete(ID,no_data_exists,axis=0)
		for key in filtration_type:
			data[key] = np.delete(data[key],no_data_exists,axis=0)
		diag = np.delete(diag,no_data_exists,axis=0)

		return ID, data, diag

	def plot_labelings(self,diag):
		'''
		create histogram summarizing the classifications
		
		inputs:
		
		diag : list of disease classifications
		
		outputs:
		
			None
		'''

		N = len(self.diagnoses_unique)        
        
		labels = np.zeros((N,))
		for i in np.arange(N):
			labels[i] = np.sum(diag==i)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.bar(self.diagnoses_unique,list(labels))
		plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

		plt.show()


	def PCA_reduction(self,data,data_type="Betti",comp=2):
		'''
		perform PCA on data
		
		inputs:
		
		data: 2d tabular array
		data_type: "Betti" for Betti curves, "PI" for persistence images
		comp: int for how many components to project onto
		
		outputs:
		
		X_vec: list of PCA-reduced data (first entry is CC, second is loops, third is CC and loops)
		'''

		def PCA_fit(X):
			pca = PCA(n_components=comp)
			mean = X.mean(axis=0)
			X_norm = X - mean                             
			X_pca = pca.fit_transform(X_norm)
			return X_pca
		if data_type == "Betti":
			data_range = 40
		elif data_type == "PI":
			data_range = 2500     
		X_vec_PCA_b0 = PCA_fit(data[:,:data_range])
		X_vec_PCA_b1 = PCA_fit(data[:,data_range:])
		X_vec_PCA_all = PCA_fit(data)

		return [X_vec_PCA_b0, X_vec_PCA_b1, X_vec_PCA_all]
    



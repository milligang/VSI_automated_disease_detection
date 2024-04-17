import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from persim import plot_diagrams

from my_persistence import *


'''
graph data file persistence
mat = np.load('My_Results/Dataset_1_output/DS1_im0001_pers_PIR.npy')
def graph(mat):
    plt.figure()
    plot_diagrams(mat)
    plt.title("1 persistence diagram")
    plt.show()
graph(mat) '''

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
		else:
			raise Exception("No diagnose file given")            
		self.diagnoses_unique = list(self.diagnosis_keys['diagnosis_codes'].values())
        
	def obtain_diagnoses(self,data_type="PI"):
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
            
		ID = np.zeros((num_images,),dtype=int)

		filtration_type = ['pers']
        
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
						file_name = (self.results_dir + self.data_name + self.file_name + num + "_" + key +"_PIR2.npy")
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


	def PCA_reduction(self,data,data_type="PI",comp=2):
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
	
	def PCA_interpretation(self, filtration):

		ID, data, diag =  self.obtain_diagnoses(data_type="PI")

		fontsize=24

		X = data[filtration][:,2500:]

		y = 1*(np.any(diag==0,axis=1))

		pca = PCA(n_components=2)

		mean = X.mean(axis=0)
		X_norm = X - mean

		X_pca = pca.fit_transform(X_norm) 

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
		ax.set_title("Flooding (loops) PCA space",fontsize=fontsize)

		ax.set_xticks([])
		ax.set_yticks([])
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

def weight_fun_ramp(x,**options):

	'''
	Weight function for persistence images

	inputs 

	x : function input
	b : max x value

	outputs 

	y: function output
	'''

	b = options.get("b")

	y = np.zeros(x.shape)

	samp = np.where(x<=0)[0]
	y[samp] = np.zeros(samp.shape)

	samp = np.where(np.logical_and(x>0,x<b))[0]
	y[samp] = x[samp]/b

	samp = np.where(x>=b)[0]
	y[samp] = np.ones(samp.shape)

	return y


def weight_fun_1(x,**options):

	'''
	Weight function of 1's for persistence images

	inputs 

	x: function input

	outputs 

	y: function output
	'''

	y = np.ones(x.shape)

	return y

def Persist_im(diag=None,filename=None,filename_save=None,inf_val=25,sigma=1e-1,weight_fun=weight_fun_ramp):

	'''
	create persistence image

	inputs

	diag :          Input Birth-death interval list. If none, then this will be loaded in from filename
	filename: 		Where Birth-death interval list is stored (leave as None if diag is not none)
	filename_save:	list of where to save persistence images: first entry is with one weighting, second with ramped weighting
	inf_val:		where to place any infinite persistence values
	sigma:			standard deviation of each Gaussian
	weight_fun:		wieght function to use to weigh each feature (based on its persistence value)    

	output

	IP : 			Persistence Image
	'''
    
	if diag is None:
		
		if filename is None:
			raise Exception("Either interval data or filename for one must be provided")
		mat = np.load(filename + '.npy',allow_pickle=True, encoding='latin1').item()

		diag = mat['BD']

	#resolution of final persistance image will be res**2
	res = 50	
	
	### Convert to non-diagonal form
	BD_list = [np.zeros((1,2)),np.zeros((1,2))]

	b0 = 0
	b1 = 0
	for dd in diag:
		if dd[0] == 0:

			if b0 == 0:
				BD_list[0][0,:] = dd[1]
			else:
				BD_list[0] = np.vstack((BD_list[0],dd[1]))
			
			b0 += 1

		elif dd[0] == 1:

			if b1 == 0:
				BD_list[1][0,:] = dd[1]
			else:
				BD_list[1] = np.vstack((BD_list[1],dd[1]))

			b1 += 1

	Ip_ones = [np.zeros((res,res)),np.zeros((res,res))]
	Ip_ramp = [np.zeros((res,res)),np.zeros((res,res))]

	for i,BD in enumerate(BD_list):

		BD[np.isinf(BD)] = inf_val
		BD_adjust = np.hstack([BD[:,0][:,np.newaxis],(BD[:,1] - BD[:,0])[:,np.newaxis]])

		width,height = np.max(BD_adjust,axis=0)
		length = inf_val#np.max((width,height))
		U = BD_adjust.shape[0]

		x = np.linspace(0,length,res+1)
		y = np.linspace(0,length,res+1)

		X,Y = np.meshgrid(x,y)

		shape = X.shape

		weights_ones = weight_fun_1(BD_adjust[:,1],b=height)
		weights_ramp = weight_fun_ramp(BD_adjust[:,1],b=height)

		for j,bd in enumerate(BD_adjust):

			Ip_tmp = np.zeros((res+1,res+1))
			for k,xx in enumerate(x):
				for l,yy in enumerate(y):
					Ip_tmp[k,l] = multivariate_normal.cdf(np.hstack((xx,yy)),
															mean=bd,
															cov=sigma)
			
			#Use summed area table (coordinates reverse of those described in wikipedia)
			Ip_ones[i] +=  weights_ones[j]*(Ip_tmp[1:,1:] + Ip_tmp[:-1,:-1] - Ip_tmp[1:,:-1] - Ip_tmp[:-1,1:])
			Ip_ramp[i] +=  weights_ramp[j]*(Ip_tmp[1:,1:] + Ip_tmp[:-1,:-1] - Ip_tmp[1:,:-1] - Ip_tmp[:-1,1:])


	if filename_save is not None:
		data = {}
		data['Ip'] = Ip_ones
		np.save(filename_save[0],data)

		data = {}
		data['Ip'] = Ip_ramp
		np.save(filename_save[1],data)

	return Ip_ones,Ip_ramp

def save_BD(diag,filename):
    
	'''
	save persistence digram

	inputs 

	diag : persistence diagram
	filename : location of where to save diag

	'''
    
	data = {}
	data['BD'] = diag
	np.save(filename,data)
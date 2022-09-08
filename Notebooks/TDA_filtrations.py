import numpy as np
import pdb, glob, time, os, imageio, copy, cv2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from ripser import ripser
from persim import plot_diagrams
import gudhi as gd
from scipy.stats import multivariate_normal
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf') 

def level_set_flooding(N,filename=None,iter_num = 50, steps = 2):
	
	'''
	level_set_flooding

	inputs:
	N : Binary image
	filename: location of where to save result (will not save if None)    
	iter_num : number of flooding events to compute
	steps: steps between recorded flooding events    

	output
	diag : Birth and death times for all topological features
	'''


	xm,ym = N.shape

	st = gd.SimplexTree()

	kernel = np.ones((3,3),np.uint8)

	for iterate in np.arange(iter_num):

		if iterate != 0:
			for j in np.arange(steps):
			    N = cv2.dilate(N,kernel,iterations=1)

		## look for vertical neighbors
		vert_neighbors = np.logical_and(N[:-1,:]==1,N[1:,:]==1)
		a = np.where(vert_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		for j in locs:
			st.insert([j,j+1],filtration = iterate)

		## look for horizontal neighbors
		horiz_neighbors = np.logical_and(N[:,:-1]==1,N[:,1:]==1)
		a = np.where(horiz_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		for j in locs:
			st.insert([j,j+xm],filtration = iterate)

		#look for diagonal neighbors (top left to bottom right)
		diag_neighbors = np.logical_and(N[:-1,:-1]==1,N[1:,1:]==1)
		a = np.where(diag_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		for j in locs:
			st.insert([j,j+xm+1],filtration = iterate)
		

		#look for diagonal neighbors (bottom left to top right)
		diag_neighbors = np.logical_and(N[1:,:-1]==1,N[:-1,1:]==1)
		a = np.where(diag_neighbors)
		a = np.hstack((a[0][:,np.newaxis],a[1][:,np.newaxis]))
		locs = a[:,0] + xm*a[:,1]
		for j in locs:
							st.insert([j+1,j+xm],filtration = iterate)

		st.set_dimension(2)

		###include 2-simplices (looking for four different types of corners)

		for j in np.arange(ym-1):
			for i in np.arange(xm-1):

				#### indices are flipped incorrectly.
				#top left corner:
				if N[i,j]==1 and N[i+1,j]==1 and N[i,j+1]==1:
					st.insert([i + xm*j,(i+1) + xm*j , i + xm*(j+1)],filtration = iterate)
				
				#top right corner
				if N[i,j]==1 and N[i+1,j]==1 and N[i+1,j+1]==1:
					st.insert([i + j*xm, (i+1)+j*xm, (i+1)  + (j+1)*xm],filtration = iterate)

				#bottom left corner
				if N[i,j]==1 and N[i,j+1]==1 and N[i+1,j+1]==1:
					st.insert([i + j*xm, i + (j+1)*xm, (i+1) + (j+1)*xm],filtration = iterate)

				#bottom right corner
				if N[i+1,j+1]==1 and N[i+1,j]==1 and N[i,j+1]==1:
					st.insert([(i+1) + (j + 1)*xm, (i+1) + j*xm, i + (j + 1)*xm],filtration = iterate)

	
	diag = st.persistence()
	if filename is not None:

		data = {}
		data['BD'] = diag
		np.save(filename,data)


	return diag

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

def image_to_pointcloud(image, threshold = 1):

	'''
	convert binary image into a pointcloud

	inputs

	image : image array
	threshold: all values above this threshold will be included in the pointcloud (all below are excluded)
    
	output

	pointcloud : list of points
	'''    
    
	pointcloud = []
	(x, y) = image.shape 
	for i in range(x):
		for j in range(y):
			if image[i,j] >= threshold:
				pointcloud.append([i,j])
	return pointcloud 

import networkx as nx
import numpy as np
#import thinning
import sys
import traceback
from collections import defaultdict
from itertools import chain
from scipy.stats import multivariate_normal
import json
import networkx as nx
import sys
import gudhi as gd

from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold

def Graph_to_weighted(graph_in):

    '''
    Graph_to_weighted : Convert unweighted graph from NEFI to weighted
    graph, where weights are euclidean distance between nodes

    inputs:

        graph_in : graph realization from NEFI

    outputs:

        graph_out : Weighted graph
    
    '''


    graph_out = nx.Graph()

    #make edges
    for n in list(graph_in.edges()):
        
        #extract two nodes from node names
        node1_str = n[0]
        node1 = tuple(map(int, node1_str[1:-1].split(', ')))
        node2_str = n[1]
        node2 = tuple(map(int, node2_str[1:-1].split(', ')))

        #determine weight via Euclidean distance
        weight = np.linalg.norm((node1[0] - node2[0],node1[1] - node2[1]))
        graph_out.add_edge(n[0],n[1],weight=weight)


    return graph_out

    
def get_nodes(graph_in):

    '''
    get_nodes : Return list of graph nodes

    inputs:

        graph_in : weighted graph

    outputs:

        nodes : list of nodes
    
    '''

    #make list of nodes
    nodes = []
    #make nodes
    for n in list(graph_in.nodes()):
        #res = tuple(map(int, n[1:-1].split(', ')))
        res = np.array((list(map(int, n[1:-1].split(', ')))))
        nodes.append(res)

    return nodes

def central_node_ID(graph_in):

    '''
    central_node_ID : Determine the most central node 
    (using betweenness centrality)

    inputs:

        graph_in : weighted graph

    outputs:

        central_node : tuple giving coordinates of most central node
    
    '''
    
    #centrality measures
    central = nx.betweenness_centrality(graph_in,weight="weight")
    
    #determine which node is the central node
    cent_max = 0
    
    #determine max centrality
    for x in central:
        if central[x] > cent_max:
            cent_max = central[x]

    #extract which nodes have max centrality
    central_node_list = []
    for x in central:
        if central[x] == cent_max:
            res = tuple(map(int, x[1:-1].split(', ')))
            central_node_list.append(res)

    #In case of multiple most central nodes -- we pick left-most
    central_node = central_node_list[0]
    for x in central_node_list:
        if x[1] < central_node[1]:
            central_node = x

    return central_node

def radius_filtration(graph_in,filename_save=None,direction='outward',max_rad=623):
    
    '''
    Compute the radial filtration

    inputs:

        graph_in : networkx graph
        filename_save : where to save data (leave as None to not save)
        direction: string, "inward"/"outward" for radial inward/outward filtration
        max_rad : max radial value to consider

    outputs:

        diag : persistence diagram
    
    '''
    
    nodes = get_nodes(graph_in)
    central_node = central_node_ID(graph_in)

    dist_from_cent = np.zeros((len(nodes)))
    for i,node in enumerate(nodes):
        dist_from_cent[i] = np.linalg.norm(node - central_node)

    st = gd.SimplexTree()

    if direction == 'outward':
        r_range = np.linspace(0,max_rad,40)
    elif direction == 'inward':
        r_range = np.linspace(max_rad,0,40)

    for i, dist in enumerate(r_range):

        if direction == 'outward':
            subgraph_ind = np.where(dist_from_cent < dist)[0]
        elif direction == 'inward':
            subgraph_ind = np.where(dist_from_cent > dist)[0]

        subgraph_nodes = []
        for j in subgraph_ind:
            subgraph_nodes.append('('+str(nodes[j][0])+', '+str(nodes[j][1])+')')

        nodes_list = list(graph_in.nodes())

        g = nx.subgraph(graph_in,subgraph_nodes)

        for node in g.nodes():
            st.insert([nodes_list.index(node)],filtration = i)
        for edge in g.edges():
            st.insert([nodes_list.index(edge[0]),nodes_list.index(edge[1])],filtration = i)

    st.set_dimension(2)

    diag= st.persistence()

    if plot == True: gd.plot_persistence_diagram(diag,legend=True)

    if filename_save is not None:
        data = {}
        data['BD'] = diag
        np.save(filename_save,data)

    return diag

def betti_curve(diag,filename_save=None,r0=0,r1=1,rN=40,plot=False):

    '''
    betti_curve construction

    inputs

        diag :          Input Birth-death interval list. If none, then this will be loaded in
        filename:       Where Birth-death interval list is stored
        filename_save:  Where to save persistence image
        r0:             Minimum value in the filtration
        r1:             Maximum value in the filtration
        rN:             Dimension of the output betti curve

    output

        b0:             CC Betti curve
        b1:             Loop Betti curve
        r_range:        parameter range over which the Betti curves were computed
    
    '''


    if diag is None:
        
        if filename is None:
            raise Exception("Either interval data or filename for one must be provided")
        mat = np.load(filename + '.npy',allow_pickle=True, encoding='latin1').item()

        diag = mat['BD']

    r_range = np.linspace(r0,r1,rN)

    b0 = np.zeros(r_range.shape)
    b1 = np.zeros(r_range.shape)

    for i,r in enumerate(r_range):
        for dd in diag:
            if r >= dd[1][0] and r < dd[1][1]:
                if dd[0] == 0:
                    b0[i] += 1
                elif dd[0] == 1:
                    b1[i] += 1

    if filename_save is not None:
        data = {}
        data['b0'] = b0
        data['b1'] = b1
        data['r'] = r_range
        np.save(filename_save,data)
        
    return b0,b1,r_range

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

	diag :          Input Birth-death interval list. If none, then this will be loaded in
	filename: 		Where Birth-death interval list is stored
	filename_save	Where to save persistence image
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

def FD_calculation(image):
    
    '''
    compute the fractal dimension

    inputs

    image:     2d binary image

    output

    FD:         Computed fractal dimension
    np.log(Ns): nparray of ln(N(s)), where s is the considered box side lengths 
    Hs:         list of images used to compute np.log(Ns)
    '''
    
    #image dimensions
    im_shape = image.shape
    im_max_shape = np.max(im_shape)

    #locations of nonzero pixels
    pixels=[]
    #will save total number of nonzero boxes for each size s
    Ns = []
    #save box-counted images for debugging
    Hs = []
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j]>0:
                pixels.append((i,j))

    Lx, Ly = im_shape
    pixels=pl.array(pixels)

    #image dimensions
    im_shape = image.shape
    im_min_shape = np.min(im_shape)

    #create logscale for s -- starts with s=1 and 
    #ends with (smaller length of image) / 2.
    scales = np.logspace(0, 
                         np.log2(im_min_shape/2), 
                         num=10, 
                         endpoint=True, 
                         base=2)
    #ensure integer valued box sizes
    scales = np.array([np.floor(s) for s in scales],dtype=int)
    #ensure 1 doesn't repeat itself
    if scales[1] == 1:
        scales[1] = 2

    
    # looping over several scales
    for scale in scales:
        
        # computing the histogram
        H, edges=np.histogramdd(pixels, 
                                bins=(np.arange(0,Lx,scale),np.arange(0,Ly,scale)))

        if scale == scales[0]:
            image_dimension_check = (H.shape[0] + 1 == im_shape[0] and H.shape[1] + 1 == im_shape[1])
            assert image_dimension_check, "Dimensions of image and boxed image of pixel 1 do not match"

        Ns.append(np.sum(H>0))
        Hs.append(H)
    # linear fit, polynomial of degree 1
    coeffs=np.polyfit(np.log(scales), np.log(Ns), 1)
    FD = -coeffs[0]
    
    return FD, np.log(Ns), Hs


def network_descriptors(graph_in):
    
    '''
    extract edges, nodes, and average edge length (pixels) from graph
    
    inputs
    
    graph_in:      network x graph
    
    outputs
    
    edges:         number of edges
    nodes:         number of nodes
    edge_lengths:  averaged length of each edge
    
    '''
    
    edges = graph_in.number_of_edges()
    nodes = graph_in.number_of_nodes()
    
    edge_length_tmp = []
    for n in list(graph_in.edges()):

        #extract two nodes from node names
        node1_str = n[0]
        node1 = tuple(map(int, node1_str[1:-1].split(', ')))
        node2_str = n[1]
        node2 = tuple(map(int, node2_str[1:-1].split(', ')))

        #determine weight via Euclidean distance
        edge_length_tmp.append(np.linalg.norm((node1[0] - node2[0],node1[1] - node2[1])))

    edge_lengths = np.mean(edge_length_tmp)
    
    return edges, nodes, edge_lengths

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

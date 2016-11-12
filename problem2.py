# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:21:14 2016

@author: yayu yao
"""

import networkx as nx
import numpy as np
import math as math
import matplotlib.pyplot as plt 

#==============================================================================
#problem 2.5
#==============================================================================

#calculate the adjacent matrix
def AdjacencyConnect(x1,x2):
    distance_sqr=pow((x1[0]-x2[0]),2)+pow((x1[1]-x2[1]),2)
    pr=1/math.sqrt(2*math.pi)*pow(math.e,-1/2*(1/8)*distance_sqr)
    if np.random.rand(1)[0]<pr:
        return True
    else :
        return False

#draw covarianceMatrix
def covarianceMatrix(cov):
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(cov, interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)        
        
#draw adjacency graph 
def adjacencyGraph(position):
    #creation of the graph
    graph = nx.Graph()
    #adding nodes/connections in the graph
    for node in range(len(pos)):
        graph.add_node(node)
    graph.add_edges_from(connect)
    
    #plot of the nodes using the (x,y) pairs as coordinates
    nx.draw(graph, [(x,y) for x,y in position],ax=None, node_size=300, with_labels=True)
    plt.axis('on')
    plt.xlim(0,1)
    plt.ylim(0,1)
       
pos = np.random.rand(20, 2) #coordinates, (x, y) for 20 nodes

#used for storing the adjacent matrix
Lamda=np.zeros((20,20))
connect = []

#create adjacent matrix according to the connection function
for i in range(len(pos)):
    for j in range(len(pos)):
        if AdjacencyConnect(pos[i],pos[j])==True:
            connect.append([i,j])
            Lamda[i][j]=0.245

for i in range(len(Lamda)):
    Lamda[i][i]=1

adjacencyGraph(pos)
plt.title('adjacency graph')
plt.show()

Sigma=np.linalg.inv(Lamda)

covarianceMatrix(Lamda)
plt.title('inverse of convariance matrix')
plt.show()

covarianceMatrix(Sigma)
plt.title('convariance matrix')
plt.show()

#==============================================================================
#problem 2.6
#==============================================================================


mean = tuple(1.5 for i in range(20))
cov = Sigma
sample = np.random.multivariate_normal(mean, cov, 1000)
#persisting the random array for debugging 
#np.savetxt("sample1000.txt", sample);
#sample_backup=np.loadtxt("sample1000.txt")
sample_covariance=np.cov(np.transpose(sample))


covarianceMatrix(sample_covariance)
plt.title('convariance matrix')
plt.show()

covarianceMatrix(np.linalg.inv(sample_covariance))
plt.title('inverse of convariance matrix')
plt.show()


#==============================================================================
#problem 2.7
#==============================================================================

#define a function to calculate the construction error used squared distance
def getReconstructionError(old,new):
    re_error=np.ones((old.shape[0],1))
    for i in range(old.shape[0]):
        for j in range(old.shape[1]):
            re_error[i]+=pow((old[i][j]-new[i][j]),2)
    
    for i in range(len(re_error)):
        re_error[i]=math.sqrt(re_error[i])
    return re_error

def drawReconstructionError(err):
    x=[i for i in range(err.shape[0])]
    y=[err[i][0] for i in range(err.shape[0])]
    plt.plot(x,y)
    
from sklearn.decomposition import PCA

    
sample_750=sample[0:750]
pca = PCA(0.95,svd_solver ='full').fit(sample_750)

components_pca =pca.components_
covarianceMatrix(components_pca)
plt.title('principal directions of PCA')
plt.show()

cov_pca=pca.get_covariance()
covarianceMatrix(cov_pca)
plt.title('estimated convariance matrix of PCA')
plt.show()

prec_pca=pca.get_precision()
covarianceMatrix(prec_pca)
plt.title('inverse convariance matrix of PCA')
plt.show()


#reconstruct the 750 points
re_750points_pca=components_pca.dot(sample_750.T)
re_750points_pca=pca.inverse_transform(re_750points_pca.T)
re_error_750_pca=getReconstructionError(sample_750,re_750points_pca)
drawReconstructionError(re_error_750_pca)
plt.title('reconstruction error of 750 points of PCA')
plt.show()  

#reconstruct the 250 points
sample_250=sample[750:1000]
re_250points_pca=components_pca.dot(sample_250.T)
re_250points_pca=pca.inverse_transform(re_250points_pca.T)
re_error_250_pca=getReconstructionError(sample_250,re_250points_pca)
drawReconstructionError(re_error_250_pca)
plt.title('reconstruction error of 250 points of PCA')
plt.show() 

#==============================================================================
#problem 2.8
#==============================================================================

def getPPCA_M():
    return np.dot(W_ppca.T,W_ppca)+Sigma_ppca*I

def getPPCA_W():
    temp=Sigma_ppca*I+np.dot(np.dot(np.dot(inv_M_ppca,W_ppca.T),S_ppca),W_ppca)
    inv_temp=np.linalg.inv(temp)
    return np.dot(np.dot(S_ppca,W_ppca),inv_temp)

def getPPCA_Sigma():
    temp=np.dot(np.dot(np.dot(S_ppca,W_ppca),inv_M_ppca),W_ppca_hat.T)
    return 1/dimension_t*np.trace(S_ppca-temp)

def getPPCACovariance():
    return W_ppca.dot(W_ppca.T) + Sigma_ppca * np.eye(dimension_x)


def getPPCAPrecision():
    return np.linalg.inv(getPPCACovariance())

def getPPCAInverseTransform(x):
    return W_ppca.T.dot(x.T) + mu_ppca
    
dimension_t=20
dimension_x=18

iterate_time=20

W_ppca=np.random.rand(20, 18)
Sigma_ppca=1
I=np.eye(dimension_x)

mu_ppca = np.mean(sample_750.T, 1)[:, np.newaxis]
S_ppca=750**-1 * (sample_750.T - mu_ppca).dot((sample_750.T - mu_ppca).T)
#S_ppca=np.cov(sample_pca.T)

for i in range(0,iterate_time):
    #at the first of iteration, calculating M and M^-1 using current W and Sigma
    M_ppca=getPPCA_M()
    inv_M_ppca=np.linalg.inv(M_ppca)
    #get the W hat
    W_ppca_hat=getPPCA_W()
    #get Sigma hat and reset the value of Sigma 
    Sigma_ppca=getPPCA_Sigma()
    #reset W
    W_ppca=W_ppca_hat

W_ppca=W_ppca.T
covarianceMatrix(W_ppca)
plt.title('principal directions of PPCA')
plt.show()

cov_ppca=getPPCACovariance()
covarianceMatrix(cov_ppca)
plt.title('estimated convariance matrix of PPCA')
plt.show()

prec_ppca=getPPCAPrecision()
covarianceMatrix(prec_ppca)
plt.title('inverse  convariance matrix of PPCA')
plt.show()


#reconstruct the 750 points
re_750points_ppca=W_ppca.dot(sample_750.T)
re_750points_ppca=getPPCAInverseTransform(re_750points_ppca.T)
re_error_750_ppca=getReconstructionError(sample_750,re_750points_ppca.T)
drawReconstructionError(re_error_750_ppca)
plt.title('reconstruction error of 750 points of PPCA')
plt.show()  

#reconstruct the 250 points
sample_250=sample[750:1000]
re_250points_ppca=W_ppca.dot(sample_250.T)
re_250points_ppca=getPPCAInverseTransform(re_250points_ppca.T)
re_error_250_ppca=getReconstructionError(sample_250,re_250points_ppca.T)
drawReconstructionError(re_error_250_ppca)
plt.title('reconstruction error of 250 points of PPCA')
plt.show() 


#==============================================================================
#problem 2.9
#==============================================================================
from sklearn.covariance import GraphLasso
gl=GraphLasso(0.01)
gl.fit(sample_750)

cov_gl=gl.covariance_
covarianceMatrix(cov_gl)
plt.title('convariance matrix of GL')
plt.show()

prec_gl=gl.get_precision()
covarianceMatrix(prec_gl)
plt.title('inverse  convariance matrix of GL')
plt.show()

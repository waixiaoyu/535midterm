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
plt.show

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
x = np.random.multivariate_normal(mean, cov, 1000)
sample_covariance=np.cov(np.transpose(x))



covarianceMatrix(sample_covariance)
plt.title('convariance matrix')
plt.show()

covarianceMatrix(np.linalg.inv(sample_covariance))
plt.title('inverse of convariance matrix')
plt.show()


#==============================================================================
#problem 2.7
#==============================================================================


from sklearn.decomposition import PCA
sample_pca=x[0:750]
pca = PCA(0.95,svd_solver ='full').fit(sample_pca)
components_pca =pca.components_
cov_pca=pca.get_covariance()
covarianceMatrix(cov_pca)
plt.title('convariance matrix')
plt.show()

prec_pca=pca.get_precision()
covarianceMatrix(prec_pca)
plt.title('inverse of convariance matrix')
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

def get_covariance():
    return W_ppca.dot(W_ppca.T) + Sigma_ppca * I


def get_precision():
    return np.linalg.inv(get_covariance())
    
dimension_t=20
dimension_x=18

iterate_time=20

W_ppca=np.random.rand(20, 18)
Sigma_ppca=1
I=np.eye(dimension_x)

#mu_ppca = np.mean(sample_pca.T, 1)[:, np.newaxis]
#S_ppca=750**-1 * (sample_pca.T - mu_ppca).dot((sample_pca.T - mu_ppca).T)
S_ppca=np.cov(sample_pca.T)

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









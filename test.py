# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:37:09 2016

@author: YangYu
"""

import numpy as np
from numpy.linalg import inv, eig

class PPCA:


    def __init__(self, n_components=2):
        self.c = n_components


    def fit(self, t, sigma=1.0, max_iter=20):
        self.t = t.T
        self.n = t.shape[0]
        self.d = t.shape[1]

        W = np.random.rand(self.d, self.c)
        mu = np.mean(t.T, 1)[:, np.newaxis]
        L = self.__cal_l(W, mu, sigma)

        S = t.shape[0]**-1 * (t.T - mu).dot((t.T - mu).T)

        for i in range(max_iter):

            M = W.T.dot(W) + sigma * np.eye(W.shape[1])
            W_new = S.dot(W).dot(inv(sigma*np.eye(W.shape[1]) + inv(M).dot(W.T).dot(S).dot(W)))
            sigma_new = self.d**-1 * np.trace(S - S.dot(W).dot(inv(M)).dot(W_new.T))

            W = W_new
            sigma = sigma_new
            L = self.__cal_l(W, mu, sigma)
            # print(i+1, round(L[0][0], 4))

        self.W = W
        self.mu = mu
        self.sigma = sigma

        return W, mu, sigma


    def __cal_l(self, W, mu, sigma):
        L = 0
        M = W.T.dot(W) + sigma * np.eye(W.shape[1])
        for i in range(self.n):
            v = self.t[:, i][:, np.newaxis]
            diff = v - mu
            xi = inv(M).dot(W.T).dot(diff)
            xixi = sigma * inv(M) + xi.dot(xi.T)
            L += 0.5 * self.d * np.log(sigma) + 0.5 * np.trace(xixi) + (2*sigma)**-1 * diff.T.dot(diff) - sigma**-1 * xi.T.dot(W.T).dot(diff) + (2*sigma)**-1 * np.trace(W.T.dot(W).dot(xixi))
        return L * -1.0


    def transform(self, t=None):
        if t == None:
            t = self.t
        else:
            t = t.T
        M = self.W.T.dot(self.W) + self.sigma * np.eye(self.W.shape[1])
        x = inv(M).dot(self.W.T).dot(t - self.mu)
        return x.T


    def inverse_transform(self, x=None):
        if x == None:
            x = self.transform().T
        else:
            x = x.T
        y = self.W.dot(x) + self.mu
        return y.T


    def get_covariance(self):
        return self.W.dot(self.W.T) + self.sigma * np.eye(self.W.shape[0])


    def get_precision(self):
        return inv(self.get_covariance())
        
        
        
        
       
        

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

class TextRank(object):
    def __init__(self, size, max_iter=100, stop_thre=1.0, damping_ratio=0.85):
        '''
            size: the size of vocab
            mat_iter: the max of iteration
            stop_thre: the distant threshold of weight change, not support yet
            damping_ratio: the shrink ratio of propagation
        '''
        self.size = size
        self.weight = np.ones(size)
        self.max_iter = max_iter
        self.stop_thre = stop_thre
        self.damping_ratio = damping_ratio

    def run(self, cnt_mat):
        '''
            cnt_mat: the frequent matrix, [i, j] from i to j 
                     and diagonal value should be 0
        '''
        (n_row, n_col) = cnt_mat.shape
        if n_row != self.size and n_col != self.size:
            raise Exception("the shape of matrix should be same with argu size")

        csr_mat = cnt_mat.tocsr()
        # normalize the row vector (that is out degree)
        norm_mat = normalize(csr_mat, norm='l1', axis=1)

        # the weight matrix
        w_mat = norm_mat.tocsc()
        for i in xrange(self.max_iter):
            # w = 0.15 + 0.85 * w * M
            self.weight = (1 - self.damping_ratio) + self.damping_ratio * self.weight * w_mat
            print self.weight
        return self.weight

if __name__ == "__main__":
    tr = TextRank(size=3, max_iter=50)
    mat = sparse.coo_matrix(([1,1, 1,1],([0,0,1,2], [1,2,0,0])), shape=(3,3))
    print mat.todense()
    tr.run(mat)
    

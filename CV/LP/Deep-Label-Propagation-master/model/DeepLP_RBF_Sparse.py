from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from model.DeepLP_RBF import DeepLP_RBF

class DeepLP_RBF_Sparse(DeepLP_RBF):

    def __init__(self, iter_, num_nodes, features, graph, sigma, lr,  print_freq=10,regularize=0,multi_class=False):
        self.phi         = tf.constant(features, dtype=tf.float32)
        self.G           = self.dense_to_sparse(tf.constant(graph, dtype=tf.float32))
        self.sigma  = tf.Variable(sigma, dtype=tf.float32)
        self.W           = self.init_weights(self.phi, self.G, self.sigma)
        self.regularize  = regularize
        self.multi_class = multi_class

        self.build_graph(iter_,lr,num_nodes)

    def dense_to_sparse(self,a_t):
        idx = tf.where(tf.not_equal(a_t, 0))
        # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
        sparse = tf.SparseTensor(idx, tf.gather_nd(a_t, idx), a_t.get_shape())
        return sparse

    def init_weights(self,phi, G, sigma):
        r = tf.reduce_sum(phi*phi, 1)
        r = tf.reshape(r, [-1, 1])
        D = tf.cast(r - 2*tf.matmul(phi, tf.transpose(phi)) + tf.transpose(r),tf.float32)
        W = tf.SparseTensor.__mul__(G,tf.exp(-tf.divide(D, sigma ** 2)))
        return W

    def forwardprop(self):
        T = self.W / tf.sparse_reduce_sum(self.W, axis = 0, keep_dims=True)
        Tnorm = T / tf.sparse_reduce_sum(T, axis = 1, keep_dims=True)

        trueX = self.X
        X = self.X

        for i in range(self.iter_):
            h = tf.transpose(tf.sparse_tensor_dense_matmul(
                            Tnorm,
                            X,
                            adjoint_a=True,
                            adjoint_b=True,
            ))
            h = tf.multiply(h, self.unlabeled) + tf.multiply(trueX, self.labeled)
            X = h
        yhat = X
        #
        #
        # def layer(i,X,trueX,Tnorm):
        #     h = tf.transpose(tf.sparse_tensor_dense_matmul(
        #         Tnorm,
        #         X,
        #         adjoint_a=True,
        #         adjoint_b=True,
        #     ))
        #
        #     h = tf.multiply(h, self.unlabeled) + tf.multiply(trueX, self.labeled)
        #     return [i+1,h,trueX,Tnorm]
        #
        #
        # def condition(i,X,trueX,Tnorm):
        #     return self.iter_ > i
        #
        # _,yhat,_,_ = tf.while_loop(condition, layer, loop_vars=[0,X,trueX,Tnorm])
        return yhat

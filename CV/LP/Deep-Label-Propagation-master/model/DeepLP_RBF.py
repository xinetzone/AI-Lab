from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from model.DeepLP import DeepLP

class DeepLP_RBF(DeepLP):

    def __init__(self, num_iter,
                       num_nodes,
                       features,
                       graph,
                       sigma,
                       lr,
                       regularize=0,       # add L1 regularization to loss
                       graph_sparse=False, # make the graph sparse
                       print_freq=10,      # print frequency when training
                       multi_class=False): # implementation for multiclass

        self.phi     = tf.constant(features, dtype=tf.float32)
        self.graph   = tf.constant(graph, dtype=tf.float32)
        self.sigma   = tf.Variable(sigma, dtype=tf.float32)
        self.weights = self._init_weights(self.phi, self.graph, self.sigma)

        self._build_graph(num_iter,
                          num_nodes,
                          lr,
                          regularize,
                          graph_sparse,
                          print_freq,
                          multi_class)

    def _save_params(self,epoch,data,n):
        sigmab = self._get_val(self.sigma)
        self.sigmas.append(sigmab)
        if epoch % 1 == 0:
            print("sigma:",sigmab)

    def _init_weights(self, phi, G, sigma):
        r = tf.reduce_sum(phi*phi, 1)
        r = tf.reshape(r, [-1, 1])
        D = tf.cast(r - 2*tf.matmul(phi, tf.transpose(phi)) + tf.transpose(r),tf.float32)
        W = tf.exp(-tf.divide(D, sigma ** 2)) * G
        return W

    def train(self,data,full_data,epochs):
        self.sigmas = []
        super().train(data,full_data,epochs)

    # def labelprop(self,data,sigma):
    #     self._open_sess()
    #     self.weights = self._init_weights(self.phi,self.graph,sigma)
    #     pred = self._eval(self.yhat,data)
    #     return pred

    def _plot_params(self):
        plt.plot(self.sigmas)
        plt.title("parameter")
        plt.show()

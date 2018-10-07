from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from model.DeepLP_RBF import DeepLP_RBF

class DeepLP_WeightedRBF(DeepLP_RBF):

    def __init__(self, num_iter,
                       num_nodes,
                       features,
                       graph,
                       sigma,
                       theta,
                       lr,
                       regularize=0,       # add L1 regularization to loss
                       graph_sparse=False, # make the graph sparse
                       print_freq=10,      # print frequency when training
                       multi_class=False): # implementation for multiclass

        self.graph    = tf.constant(graph, dtype=tf.float32)
        self.sigma    = tf.constant(sigma, dtype=tf.float32)
        self.theta    = tf.Variable(tf.convert_to_tensor(theta, dtype=tf.float32))
        self.features = tf.constant(features, dtype=tf.float32)
        self.phi      = self.features * self.theta
        self.weights  = self._init_weights(self.phi, self.graph, self.sigma)

        self._build_graph(num_iter,
                          num_nodes,
                          lr,
                          regularize,
                          graph_sparse,
                          print_freq,
                          multi_class)

    def _save_params(self,epoch,data,n):
        thetab = self._get_val(self.theta)
        self.thetas.append(thetab)
        if epoch % 1 == 0:
            print("theta:",thetab)

    def train(self,data,full_data,epochs):
        self.thetas = []
        super().train(data,full_data,epochs)

    # def labelprop(self,data,theta):
    #     self._open_sess()
    #     self.theta   = tf.Variable(tf.convert_to_tensor(theta, dtype=tf.float32))
    #     self.phi     = self.features * self.theta
    #     self.weights = self._init_weights(self.phi, self.graph, self.sigma)
    #     pred = self._eval(self.yhat,data)
    #     return pred

    def _plot_params(self):
        plt.plot(self.thetas)
        plt.title("parameters")
        plt.show()

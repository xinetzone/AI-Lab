from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from model.DeepLP_RBF_Sparse import DeepLP_RBF_Sparse

class DeepLP_WeightedRBF_Sparse(DeepLP_RBF_Sparse):


    def __init__(self, iter_, num_nodes, features, graph, sigma_, theta_, lr, regularize=0, multi_class=False):
        phi          = tf.constant(features, dtype=tf.float32)
        G            = self.dense_to_sparse(tf.constant(graph, dtype=tf.float32))
        self.sigma   = tf.constant(sigma_, dtype=tf.float32)
        num_features = features.shape[1]
        self.theta   = tf.Variable(tf.convert_to_tensor(theta_, dtype=tf.float32))
        phi          = phi * self.theta
        self.W       = self.init_weights(phi, G, sigma_)
        self.regularize = regularize
        self.multi_class = multi_class

        self.build_graph(iter_,lr,num_nodes)

    def save_params(self,epoch,data,n):
        thetab = self.get_val(self.theta)
        self.thetas.append(thetab)
        if epoch % 10 == 0:
            print("theta:",thetab)

    def train(self,data,full_data,epochs):
        self.thetas = []
        super().train(data,full_data,epochs)

    def plot_params(self):
        plt.plot(self.thetas)
        plt.title("parameters")
        plt.show()

import sklearn
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from model.utils import indices_to_vec

class Data:
    '''
    Load datasets.
    '''
    def load_iris():
        # load iris data
        iris   = sklearn.datasets.load_iris()
        data   = iris["data"]
        labels = iris["target"]

        # get label 0 and 1, and corresponding features
        true_labels = labels[labels < 2]
        features = data[np.where(labels < 2)]

        return true_labels, features, []

    def load_cora(multiclass=False):
        # load cora data
        if multiclass:
            nodes = pd.read_csv('cora/selected_contents_multiclass.csv',index_col=0,)
        else:
            nodes = pd.read_csv('cora/selected_contents.csv',index_col=0,)
        graph = np.loadtxt('cora/graph.csv',delimiter=',')
        id_    = np.array(nodes.index)

        # get label 0 and 1, and corresponding features
        true_labels = np.array(nodes['label'])
        features   = nodes.loc[:,'feature_0':].as_matrix()

        return true_labels, features, graph

    def prepare(labels,labeled_indices,true_labels):
        num_nodes = len(labels)
        X_ = np.tile(labels,(len(labeled_indices),1))
        y_ = true_labels.reshape((1,len(true_labels)))
        true_labeled_ = indices_to_vec(labeled_indices,num_nodes).reshape((1,len(true_labels)))
        labeled_ = np.tile(true_labeled_,(len(labeled_indices),1))
        masked_  = np.zeros((len(labeled_indices),num_nodes))

        validation_data = {
            'X': labels.reshape(1,num_nodes),
            'y': y_,
            'labeled': true_labeled_,
            'true_labeled': true_labeled_.reshape(1,num_nodes), # this will not be used
            'masked': masked_  # this will not be used
        }

        for i,labeled_index in enumerate(labeled_indices):
            X_[i,labeled_index] = 0.5
            labeled_[i,labeled_index] = 0
            masked_[i,labeled_index] = 1

        data = {
            'X': X_,
            'y': y_,
            'labeled': labeled_,
            'true_labeled': true_labeled_.reshape(1,num_nodes),
            'masked': masked_
        }

        return data, validation_data

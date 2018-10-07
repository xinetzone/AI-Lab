from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import argparse
import sys
sys.path.append('../')
from model import *

def main(args):

    if args.data == "cora":
        true_labels, features, graph = Data.Data.load_cora(rel_path='../')
        if not args.graph:
            graph = []

    np.random.seed(args.seed)
    sigma = np.random.uniform(0.2,10)
    theta = np.random.uniform(0.2,10,(1,features.shape[1]))

    labels, is_labeled, labeled_indices, unlabeled_indices \
    = utils.random_unlabel(true_labels,
                           unlabel_prob=1-float(args.labeled_percentage),
                           hard=False,seed=args.seed)
    solution  = true_labels[unlabeled_indices]

    weights, graph = utils.rbf_kernel(features,
                                      s=sigma,
                                      G=graph,
                                      percentile=args.threshold_percentage)
    num_nodes = len(labels)
    data, validation_data = Data.Data.prepare(labels,labeled_indices,true_labels)

    if args.link_function == "rbf":

        dlp_rbf = DeepLP_RBF.DeepLP_RBF(num_nodes,
                                        features,
                                        graph,
                                        sigma,
                                        num_iter=args.num_iter,
                                        lr=args.lr,
                                        regularize=args.regularize,
                                        loss_type=args.loss)
        dlp_rbf.train(data,validation_data,args.num_epoch)

    if args.link_function == "wrbf":

        dlp_wrbf = DeepLP_WeightedRBF.DeepLP_WeightedRBF(num_nodes,
                                                         features,
                                                         graph,
                                                         sigma,
                                                         theta,
                                                         num_iter=args.num_iter,
                                                         lr=args.lr,
                                                         regularize=args.regularize,
                                                         loss_type=args.loss)
        dlp_wrbf_prediction,_ = dlp_wrbf.labelprop(validation_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",                 default="cora")
    parser.add_argument("--graph",                default=0)
    parser.add_argument("--k",                    default=1)
    parser.add_argument("--labeled_percentage",   default=0.1)
    parser.add_argument("--link_function",        default="rbf", help="rbf, wrbf")
    parser.add_argument("--loss",                 default="mse", help="mse, log")
    parser.add_argument("--lr",                   default=0.1)
    parser.add_argument("--num_epoch",            default=1000)
    parser.add_argument("--num_iter",             default=100)
    parser.add_argument("--regularize",       default=0)
    parser.add_argument("--seed",                 default=None)
    parser.add_argument("--threshold_percentage", default=0)

    args = parser.parse_args()
    main(args)

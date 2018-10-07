import numpy as np
from numpy.linalg import inv

import sys
sys.path.append('../')
from model.utils import rbf_kernel

class LP:
    '''
    Label propagation for predicting labels for unlabeled nodes.
    Closed form and iterated solutions.
    See mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf for details.
    '''
    def __init__(self):
        return

    def closed(self, labels,
                     weights,
                     labeled_indices,
                     unlabeled_indices):
        '''
        Closed solution of label propagation.
        '''
        # normalize T
        Tnorm = self._tnorm(weights)
        # sort Tnorm by unlabeled/labeld
        Tuu_norm = Tnorm[np.ix_(unlabeled_indices,unlabeled_indices)]
        Tul_norm = Tnorm[np.ix_(unlabeled_indices,labeled_indices)]
        # closed form prediction for unlabeled nodes
        lapliacian = (np.identity(len(Tuu_norm))-Tuu_norm)
        propagated = Tul_norm @ labels[labeled_indices]
        label_predictions = np.linalg.solve(lapliacian, propagated)
        return label_predictions

    def iter(self, X, # input labels
                   weights,
                   labeled,
                   num_iter):
        '''
        Iterated solution of label propagation.
        '''
        # normalize T
        Tnorm = self._tnorm(weights)
        h = X.copy()

        for i in range(num_iter):
            # propagate labels
            h = np.dot(h,Tnorm.T)
            # don't update labeled nodes
            h = h * (1-labeled) + X * labeled

        # only return label predictions
        return h

    def iter_multiclass(self,X, # input labels
                              weights,
                              labeled_indices,
                              unlabeled_indices,
                              num_iter=-1):
        preds = []
        num_classes = len(set(X))
        for class_ in range(num_classes):
            X_class = X.copy()
            X_class[labeled_indices] = X_class[labeled_indices] == class_
            X_class[unlabeled_indices] = np.array([1/num_classes] * len(unlabeled_indices))
            if num_iter == -1:
                pred = closed(X_class,weights,labeled_indices,unlabeled_indices)
            else:
                pred = iter(X_class,weights,labeled_indices,unlabeled_indices,iter_)
            preds.append(pred)
        res = np.vstack(preds).T
        return res

    def _tnorm(self,weights):
        '''
        Column normalize -> row normalize weights.
        '''
        # row normalize T
        Tnorm = weights / np.sum(weights, axis=1, keepdims=True)
        return Tnorm

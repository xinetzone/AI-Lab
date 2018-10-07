from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import time

class DeepLP:
    '''
    Deep label propagation for predicting labels for unlabeled nodes.
    See our paper for details.
    '''
    def __init__(self, num_iter,
                       num_nodes,
                       weights,
                       lr,
                       regularize=0,       # add L1 regularization to loss
                       graph_sparse=False, # make the graph sparse
                       print_freq=10,      # print frequency when training
                       multi_class=False): # implementation for multiclass
        self.weights      = self._init_weights(weights)

        self._build_graph(num_iter,
                          num_nodes,
                          lr,
                          regularize,
                          graph_sparse,
                          print_freq,
                          multi_class)

    def _build_graph(self, num_iter,
                           num_nodes,
                           lr,
                           regularize,
                           graph_sparse,
                           print_freq,
                           multi_class):
        self.start_time   = time.time()

        # set instance variables
        self.num_iter     = num_iter
        self.lr           = lr
        self.regularize   = regularize
        self.graph_sparse = graph_sparse
        self.print_freq   = print_freq
        self.multi_class  = multi_class

        # initialize placeholders
        shape             = [None, num_nodes]
        self.X            = tf.placeholder("float", shape=shape)
        self.y            = tf.placeholder("float", shape=shape)
        self.labeled      = tf.placeholder("float", shape=shape)
        self.true_labeled = tf.placeholder("float", shape=shape)
        self.masked       = tf.placeholder("float", shape=shape)

        self.yhat         = self._forwardprop(self.X,
                                              self.weights,
                                              self.labeled,
                                              self.num_iter)
        self.update, self.metrics   = self._backwardprop(self.y,
                                                         self.yhat,
                                                         self.labeled,
                                                         self.true_labeled,
                                                         self.masked,
                                                         self.regularize,
                                                         self.lr)

    def _get_val(self, val):
        return self.sess.run(val)

    def _init_weights(self, weights_np):
        """ Weight initialization. """
        weights = tf.convert_to_tensor(weights_np, np.float32)
        return tf.Variable(weights)

    def _tnorm(self, weights):
        Tnorm = weights / tf.reduce_sum(weights, axis = 1, keep_dims=True)
        return Tnorm

    def _forwardprop(self, X,
                           weights,
                           labeled,
                           num_iter):
        '''
        Forward prop which mimicks LP.
        '''

        Tnorm = self._tnorm(weights)

        def layer(i,h,X,Tnorm):
            # propagate labels
            h = tf.matmul(h,Tnorm,transpose_b=True)
            # don't update labeled nodes
            h = tf.multiply(h, (1-labeled)) + tf.multiply(X, labeled)
            return [i+1,h,X,Tnorm]

        def condition(i,X,trueX,Tnorm):
            return num_iter > i

        _,yhat,_,_ = tf.while_loop(condition, layer, loop_vars=[0,X,X,Tnorm])
        return yhat

    def _regularize_loss(self):
        return 0
        # tf.nn.l2_loss(self.theta-1)

    def _backwardprop(self, y,
                            yhat,
                            labeled,
                            true_labeled,
                            masked,
                            regularize,
                            lr):
        '''
        Backprop on unlabeled + masked labeled nodes.
        Calculate loss and accuracy for both train and validation dataset.
        '''
        # backward propagation
        l_o_loss      = (self._calc_loss(y,yhat,masked)
                              + regularize * self._regularize_loss())
        update        = tf.train.AdamOptimizer(lr).minimize(l_o_loss)

        # evaluate performance
        loss          = self._calc_loss(y,yhat,1-labeled)
        # labeled_loss  = self._calc_loss(y,yhat,labeled)
        accuracy      = self._calc_accuracy(y,yhat,1-labeled)
        true_loss     = self._calc_loss(y,yhat,1-true_labeled)
        true_accuracy = self._calc_accuracy(y,yhat,1-true_labeled)

        metrics = {
            'loss': loss,
            'labeled_loss': l_o_loss,
            'accuracy': accuracy,
            'true_loss': true_loss,
            'true_accuracy': true_accuracy
        }

        return update, metrics

    def _calc_loss(self,y,yhat,mask):
        loss_mat = tf.multiply(mask, (y-yhat) ** 2 )
        return tf.reduce_sum(loss_mat) / tf.count_nonzero(mask,dtype=tf.float32)

    def _calc_accuracy(self,y,yhat,mask,true_data=False):
        acc_mat = tf.multiply(mask,tf.cast(tf.equal(tf.round(yhat),y),tf.float32))
        if true_data:
            return tf.reduce_sum(acc_mat) / tf.count_nonzero(mask,dtype=tf.float32)
        else:
            return tf.reduce_sum(acc_mat) / tf.count_nonzero(mask,dtype=tf.float32)

    def _open_sess(self):
        self.sess = tf.Session()
        init      = tf.global_variables_initializer()
        self.sess.run(init)

    def labelprop(self,data):
        self._open_sess()
        l_o_loss      = (self._calc_loss(self.y,self.yhat,self.masked)
                              + self.regularize * self._regularize_loss())
        # evaluate performance
        loss          = self._calc_loss(self.y,self.yhat,1-self.labeled)
        # labeled_loss  = self._calc_loss(y,yhat,labeled)
        accuracy      = self._calc_accuracy(self.y,self.yhat,1-self.labeled)
        true_loss     = self._calc_loss(self.y,self.yhat,1-self.true_labeled)
        true_accuracy = self._calc_accuracy(self.y,self.yhat,1-self.true_labeled)

        pred,l_o_loss_,loss_,accuracy_,true_loss_,true_accuracy_ = self._eval([self.yhat,l_o_loss,loss,accuracy,true_loss,true_accuracy],data)
        metrics = {
            'loss': loss_,
            'labeled_loss': l_o_loss_,
            'accuracy': accuracy_,
            'true_loss': true_loss_,
            'true_accuracy': true_accuracy_
        }
        print(metrics)

        return pred, metrics

    def _eval(self,vals,data):
        return self.sess.run(vals, feed_dict={self.X:data['X'],
                                              self.y:data['y'],
                                              self.labeled:data['labeled'],
                                              self.true_labeled:data['true_labeled'],
                                              self.masked:data['masked']})

    def _save_params(self,epoch,data,n):
        pass

    def _save(self,epoch,data,validation_data,n):

        loss,labeled_loss,accuracy = self._eval([self.metrics['loss'],self.metrics['labeled_loss'],self.metrics['accuracy']],data)
        true_loss,true_accuracy    = self._eval([self.metrics['true_loss'],self.metrics['true_accuracy']],validation_data)
        self.losses.append(loss)
        self.labeled_losses.append(labeled_loss)
        self.accuracies.append(accuracy)
        self.true_losses.append(true_loss)
        self.true_accuracies.append(true_accuracy)

        if epoch % 1 == 0 or epoch == -1:
            print("epoch:",epoch,
                  "labeled loss:",labeled_loss,
                  "unlabeled loss:",loss,
                  "accuracy:",accuracy,
                  "true unlabeled loss:",true_loss,
                  "true accuracy:",true_accuracy)
            print("--- %s seconds ---" % (time.time() - self.start_time))
            self.start_time = time.time()
        self._save_params(epoch,data,n)

    def train(self,data,validation_data,epochs):
        self._open_sess()

        n = len(data['X'])
        self.losses = []
        self.labeled_losses = []
        self.accuracies = []
        self.true_losses = []
        self.true_accuracies = []
        self._save(-1,data,validation_data,n)
        for epoch in range(epochs):
            # Train with each example
            self._eval(self.update,data)
            print("--- %s seconds ---" % (time.time() - self.start_time))
            self.start_time = time.time()
            self._save(epoch,data,validation_data,n)

    def _plot_loss(self):
        plt.plot(self.labeled_losses,label="labeled loss")
        plt.plot(self.losses,label="unlabeled loss")
        plt.plot(self.true_losses,label='validation unlabeled loss')
        plt.title("loss")
        plt.legend()
        plt.show()

    def _plot_accuracy(self):
        plt.plot(self.accuracies,label="DeepLP, train")
        plt.plot([self.true_accuracies[0]] * len(self.accuracies),label="LP")
        plt.plot(self.true_accuracies,label="DeepLP, validation")
        plt.title("accuracy")
        plt.legend()
        plt.show()

    def _plot_params(self):
        pass

    def plot(self):
        self._plot_loss()
        self._plot_accuracy()
        self._plot_params()

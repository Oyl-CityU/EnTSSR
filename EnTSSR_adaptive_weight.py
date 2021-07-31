#!/usr/bin/python
#coding:utf-8

import os
import numpy as np

seed=123
# np.random.seed(seed)
# import tensorflow as tf
# tf.set_random_seed(seed)

import numba as nb
import pandas as pd
import keras
import numpy.linalg as LA

from memory_profiler import profile
from scipy import stats
from functions import *
import time
os.environ['R_HOME'] = '/home/anaconda3/envs/r37/lib/R'# Set the environment variable 'R_HOME' to the path where R is located
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

FloatType='float64'
input_dir = os.path.join(os.path.join(os.curdir,"data"),"input")
res_dir = os.path.join(os.path.join(os.curdir,"data"),"res")

class EnTSSR_adaptive_weight:
    '''
    class of EnTSSR
    '''
    def __init__(self,dataset, beta=1, gamma=0.25, max_iter=10, epoch=20):
        '''
        initialize parameter
        '''
        self.beta = beta # L1 regularization
        self.gamma = gamma #entropy regularization
        self.max_iter = max_iter # max iteration
        self.epoch = epoch # the number of epochs in Keras
        self.dataset = dataset 
        self.methods = np.array(["ALRA", "DCA", "MAGIC","SAVER", "scImpute", "scRMD"])#base imputation methods 

    def objective_value(self):
        '''return the value of objective function'''
        tssr_error = 0
        for i in range(len(self.methods)):
            ensemble_term = self.predMat[i] - self.gene_sim.dot(self.predMat[i]) - self.predMat[i].dot(self.cell_sim.T)\
                            - self.gene_sim.dot(self.predMat[i]).dot(self.cell_sim.T)
            tssr_error = tssr_error + self.wi[i] * LA.norm(ensemble_term, 'fro') ** 2
        L1_error = self.beta * (LA.norm(self.gene_sim, 1) + LA.norm(self.cell_sim, 1))
        wi_error = self.gamma * np.size(self.Y_res_obs) * self.wi.dot(np.log(self.wi))
        obj_val = tssr_error + L1_error + wi_error
        return obj_val, tssr_error, L1_error, wi_error

    def keras_lasso_regression(self,Y, X, initG, epochs = 100, batch_size = 128, beta = 0.01, lambda2 = 1e10,
                                    lasso_threshold = 0, seed = None, verbose = True):
        '''solve Gg and Gc using Keras'''
        import tensorflow as tf
        from keras import backend as K
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.constraints import non_neg
        from keras.initializers import random_uniform,random_normal,zeros
        from keras.utils import multi_gpu_model
        from keras import optimizers

        def loss_define(y_pred, y_true):
            return K.sum(K.square(y_true - y_pred))
        def reg_define(weight_matrix, beta = beta, lambda2 = 1e10):
            return beta * K.sum(K.abs(weight_matrix)) + lambda2 * tf.linalg.trace(tf.square(weight_matrix))
        def init_define(shape,dtype=None):
            return K.constant(initG, shape = shape,dtype = dtype)


        n,p = X.shape
        model = Sequential()
        model.add(Dense(units=p, input_shape=(p,), activation='linear',
                        kernel_regularizer = reg_define,
                        use_bias=False, kernel_constraint=non_neg(), kernel_initializer=init_define)) #random_normal(stddev=1/p,seed=seed))),random_uniform(minval=0,maxval=(1/p))
        adam = optimizers.Adam(learning_rate=0.001) #default 0.001
        model.compile(loss=loss_define,
                      optimizer=adam,
                      metrics=['mean_squared_error'])
        model.fit(x = X, y = Y, epochs = epochs, batch_size = batch_size, validation_split = 0,
                                    verbose = verbose)
        weight = model.get_weights()[0]
        weight[weight<=lasso_threshold] = 0
        K.clear_session()
        return weight

    def fix_model(self, dataset: str, Y, seed: int=None):
        '''
        Update variables W,Gg and Gc. 
        '''
        def cal_P_Q(optim="Gc"):
            P = np.array([], dtype=FloatType)
            Q = P
            if optim == 'Gc':
                ##
                P = np.concatenate([np.sqrt(self.wi[i]) * (self.predMat[i] - self.gene_sim.dot(self.predMat[i]))
                                    for i in range(self.n_methods)], axis=0)
                Q = np.concatenate([np.sqrt(self.wi[i]) * (self.predMat[i] + self.gene_sim.dot(self.predMat[i]))
                                    for i in range(self.n_methods)], axis=0)
            elif optim == 'Gg':
                P = np.concatenate([np.sqrt(self.wi[i]) * (self.predMat[i].T - self.cell_sim.dot(self.predMat[i].T))
                                    for i in range(self.n_methods)], axis=0)
                Q = np.concatenate([np.sqrt(self.wi[i]) * (self.predMat[i].T + self.cell_sim.dot(self.predMat[i].T))
                                    for i in range(self.n_methods)], axis=0)
            return P, Q


        ### load dataset
        ## 1）Load observed data matrix; 2）Load base impuation methods
        self.dataset=dataset
        self.predMat = read_PredMats(dataset, self.methods, rescale=True) 
        self.n_methods, self.n_genes, self.n_cells = self.predMat.shape

        self.Y_exp_obs = np.array(pd.read_csv(os.path.join(input_dir, dataset + "_exp_obs.csv"),index_col= 0))
        self.cell_names, self.gene_names = get_genes_cells_names(dataset)

        self.Y_res_obs = Y.astype(FloatType).copy()# 
        self.Y_preImpute = self.Y_res_obs.copy()
        ## calculate pre-imputed matrix
        self.Y_preImpute[self.Y_res_obs == 0] = stats.trim_mean(self.predMat, proportiontocut=0.3, axis=0)[self.Y_res_obs == 0]  

        ## Initialize the variable W, Gg and Gc.
        self.wi = 1 / self.n_methods * np.ones((self.n_methods,))
        self.gene_sim = np.zeros(shape=(self.n_genes, self.n_genes)).astype(FloatType)
        self.cell_sim = np.zeros(shape=(self.n_cells, self.n_cells)).astype(FloatType)#self.Sc#

        tic1 = time.time()
        save_loss, save_delta_loss = [], []
        last_loss, _, _, _ = self.objective_value()
        save_loss.append(last_loss);save_delta_loss.append(np.nan)

        self.wi_allIter = -np.ones((self.max_iter,self.n_methods))#Record the weights of each iteraction.
        ## Update
        for t in range(self.max_iter):
            ## update of similarity between cells: Gc
            P, Q = cal_P_Q('Gc')
            old_cell_sim = self.cell_sim
            self.cell_sim = self.keras_lasso_regression(Y = P, X = Q,initG= self.cell_sim, beta=self.beta, batch_size = 512,
                                                        epochs = self.epoch, seed=seed, verbose=False).T
            if t>0:
                self.cell_sim = (self.cell_sim + old_cell_sim)/2.0 
            ## update of similarity between genes: Gg
            P, Q = cal_P_Q('Gg')
            old_gene_sim = self.gene_sim
            self.gene_sim = self.keras_lasso_regression(Y = P, X = Q,initG=self.gene_sim, beta=self.beta, batch_size = 512,
                                                        epochs = self.epoch, seed=seed, verbose=False).T
            if t>0:
                self.gene_sim = (self.gene_sim + old_gene_sim)/2.0

            ## update of weight
            weight=np.zeros(shape=np.shape(self.wi))
            for i in np.arange(self.n_methods):
                ensemble_term = self.predMat[i] - self.gene_sim.dot(self.predMat[i]) - self.predMat[i].dot(
                    self.cell_sim.T) - self.gene_sim.dot(self.predMat[i]).dot(self.cell_sim.T)
                weight[i] = np.exp(-1 / self.gamma / np.size(self.Y_res_obs) * LA.norm(ensemble_term,"fro") ** 2)
            self.wi = weight / np.sum(weight)
           
            ## stop until converge
            self.wi_allIter[t,:] = self.wi
            cur_loss, tssr_error, L1_error, wi_error = self.objective_value()
            delta_loss = (cur_loss - last_loss)/abs(last_loss)
            save_loss.append(cur_loss); save_delta_loss.append(delta_loss)
            last_loss = cur_loss
            print(("%s/%s\tloss:%s\tdelta_loss:%s\ttssr_error:%s\tL1_error:%s\twi_error:%s")%
                  (t+1, self.max_iter, last_loss, delta_loss,
                    tssr_error, L1_error, wi_error))
            if abs(delta_loss) < 1e-5 and t > 0:
                self.wi_allIter = self.wi_allIter[:t+1,:]
                break
        self.loss = cur_loss
        for t in range(len(save_loss)):
            print(("%s/%s\tloss:%s\tdelta_loss:%s") %
                  (t + 1, self.max_iter, save_loss[t], save_delta_loss[t]))
        print("Ensemble:%s" % self.methods)
        print("wi:%s" % self.wi)
        print("- Non-zero rate of gene sim:%s" % (len(np.argwhere(self.gene_sim > 1e-8)) / np.size(self.gene_sim)))
        print("- Non-zero rate of cell sim:%s" % (len(np.argwhere(self.cell_sim > 1e-8)) / np.size(self.cell_sim)))
        plotloss(np.array(save_loss),np.array(save_delta_loss),title=self.__str__(),dataset=dataset) #Plotting the change of the objective function value
 

 
        ## Reconstructing：Gg*Y*Gc'+ Gg*Y + Y*Gc'
        self.impute_log = self.gene_sim.dot(self.Y_preImpute) + self.Y_preImpute.dot(self.cell_sim.T)\
                           + self.gene_sim.dot(self.Y_preImpute).dot(self.cell_sim.T)
 
        self.impute_log[self.impute_log < 0] = 0
        self.impute_exp = np.exp(self.impute_log) - 1# convert to a linear scale

        ## postprocessed by SAVER
        rpy2.robjects.numpy2ri.activate()
        a = ro.r.matrix(self.impute_exp, np.shape(self.impute_exp)[0],np.shape(self.impute_exp)[1])
        ro.r.assign("A", a)
        b = ro.r.matrix(self.Y_exp_obs, np.shape(self.Y_exp_obs)[0],np.shape(self.Y_exp_obs)[1])
        ro.r.assign("B", b)
        ro.r.assign("seed", seed)
        ro.r("print(seed)")
        ro.r("set.seed(seed)")
        ro.r("SAVER::saver(B,ncores=30, mu = A)$estimate")#
        c = ro.r("c=SAVER::saver(B, mu = A)$estimate")#
        ro.r('rm(list=ls())')
        self.impute_exp = np.array(list(c))## The final imputation results of EnTSSR

    def impute_result(self):
        return self.impute_exp


    def __str__(self):
        cmd = ("method:EnTSSR_adaptive_weight, beta=%s,gamma=%s, max_iter=%s,epoch=%s")%\
              ( self.beta,self.gamma, self.max_iter, self.epoch)
        return cmd



#######################################################
#######################################################
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    set_session(tf.Session(config=config))

    datasets = ["baron","manno","chen","zeisel","guo2","pollen","iPSC","loh","usoskin","deng","chu"]#
    this_method = "EnTSSR"
    ari_list,corr_list,auc_list = [],[],[]
    seed = 123

    for dataset in datasets:
        ## random seed
        np.random.seed(seed)
        import tensorflow as tf
        tf.set_random_seed(seed)
        ##
        Y = np.array(pd.read_csv(os.path.join(input_dir, dataset + "_res_obs.csv"), index_col=0))

        tic1 = time.time()
        model = EnTSSR_adaptive_weight(dataset, beta=1, gamma=2, max_iter=10, epoch=20)# setting parameter of EnTSSR
        print("Dataset:%s\n" % dataset)
        print(model.__str__())
        model.fix_model(dataset, Y, seed = seed)

        gene_names, cell_names = get_genes_cells_names(dataset)
        df = pd.DataFrame(model.impute_exp,index=gene_names,columns=cell_names)#save 
        df_wi = pd.DataFrame(model.wi_allIter,index=np.arange(1,model.wi_allIter.shape[0]+1),columns=model.methods)
        # save the imputation results and weight
        try:
            result_path = os.path.join(res_dir, dataset + "_exp_" + this_method + ".csv")
            wi_path = os.path.join(res_dir,"weight_"+ dataset + "_" + this_method + ".csv")
            df.to_csv(result_path) # save the imputation result
            df_wi.to_csv(wi_path) #save weight
            print("save impute result successfully! Save at:",result_path)
            print("save weight successfully! Save at:",wi_path)
        except:
            print("Fail to save impute results!")
        print("Dataset:%s\n" % dataset)
        print("Ensemble:%s" % model.methods)
        print("wi:%s" % model.wi)


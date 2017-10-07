#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 14:26:33 2017

@author: charleshen
"""
import pandas as pd
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix,roc_curve,auc

from sklearn.metrics import cohen_kappa_score,matthews_corrcoef, make_scorer

from sklearn.grid_search import GridSearchCV

#import post prune tree method
from sklearn.tree import tree_prune

from copy import deepcopy
from IPython.display import Image
from operator import itemgetter
from time import time
import pydotplus
import random


'''
class DTGetBestParas(object):
    def __init__(self, **kwargs):
        
'''

def get_min_samples(N):
    '''
    N ---   number of the samples
    return
            number of min_samples_split , min_samples_leaf
    '''
    if N>=1000000000:
        return 500, 250
    else:
        if N >=1000000:
            s = min(0.0000002*N+200,400)
            return int(s),int(s/2)
        else:
            if N>=10000:
                s = min(0.00015*N+50,200)
                return int(s),int(s/2)
            else:
                return 50,25



def report(grid_scores, n_top=3):
    '''
    Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    '''
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters


def run_gridsearch(X, y, clf, param_grid, cv=5, n_jobs = 1,
                   scoring = make_scorer(matthews_corrcoef)):
    '''
    Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    '''
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               scoring = scoring,
                               cv=cv,n_jobs = n_jobs)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return  top_params



def tune_paras(clf,X,y, random_state = 100,cv =5,scoring = 'roc_auc',n_jobs =1):
    
    min_samples_split,min_samples_leaf = get_min_samples(len(y))
    clf.min_samples_split = int(0.5*min_samples_split)
    clf.min_samples_leaf = int(0.5*min_samples_leaf)
    clf.fit(X,y)        
    depth = tree_prune.get_max_depth(clf)
    
    if depth >=10:
        depth_ls = list(range(10,depth,2))
    else:
        depth_ls = [4,6,8,10]
        
    depth_ls.append(None)
    
    #tune dist
    param_dist = {'max_depth': depth_ls}
    seed_list = [random_state,17]
    
    dfall =pd.DataFrame()
    for i in seed_list:  
        clf.random_state = i
        clf.fit(X,y)
        ts_gs = run_gridsearch(X, y, clf, 
                               param_grid =param_dist,
                               cv=cv, scoring = scoring,
                               n_jobs = n_jobs)
        ts_gs['min_samples_split'] = clf.min_samples_split
        ts_gs['min_samples_leaf'] = clf.min_samples_leaf
        
        df = pd.DataFrame.from_dict(ts_gs,orient = 'index').T
        df.index = [i]
        dfall = dfall.append(df)
    
    paras = pd.DataFrame(mode(dfall)[0],columns = dfall.columns).T[0].to_dict()
    return paras



def get_optimal_leaves(clf,X_train,y_train,n_iterations=5,random_state = 100,alpha = 0.5):
    
    leaves = tree_prune.get_n_leaves(clf)
    
    
    if leaves <= 500:
        pass
    else:
        leaves = 500
        
    scores = tree_prune.prune_path(clf, 
                             X_train,
                             y_train,
                             max_n_leaves=int(leaves*alpha), 
                             n_iterations=n_iterations,
                             random_state=random_state)    

    score = list(scores)
    means = np.array([np.mean(s) for s in score])
    stds = np.array([np.std(s) for s in score]) / np.sqrt(len(score[1]))
    
    x =range(int(leaves*alpha),1,-1)
    
    
    dfp = pd.DataFrame(means+stds,index = x)
    
    
    return max(dfp.idxmax()),means,stds,x


def plot_prune(x,means,stds):
    plt.figure(figsize=(16,8))
    import matplotlib 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    plt.plot(x, means,lw = 3)
    plt.plot(x, means + 2 * stds, lw=3, c='0.7')
    plt.plot(x, means - 2 * stds, lw=3, c='0.7')
    plt.xlabel('Number of leaf nodes',fontsize = 24)
    plt.ylabel('Cross validated score',fontsize = 24)
    plt.show()

     

def get_tree_paras(df, method = 'none', n_iter = 10, cv = 5, n_jobs = 1,random_state = 100, alpha = 0.5):
    '''
    input:
        df: dataframe, last column is target(label) 
        
        method: can be 'none','cal','tune','prune','both'
        
    return:
        paras: dict
        
        clf: tree object
        
    '''
    
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    
        
    if len(np.unique(y)) <= 10:
        clf = tree.DecisionTreeClassifier()  
        clf_prune = tree_prune.DecisionTreeClassifier()
        scoring = 'roc_auc'
    else:
        clf = tree.DecisionTreeRegressor() 
        clf_prune = tree_prune.DecisionTreeRegressor() 
        scoring = 'r2'
    clf.random_state = random_state
    clf_prune.random_state = random_state
    
    clf.fit(X,y)
    depth = tree_prune.get_max_depth(clf)
    leaf_samples,node_samples = tree_prune.get_min_sample_leaf_split(clf)
    
    if method == 'none':
        paras = {}
        paras['max_depth'] = depth
        paras['min_samples_split'] = node_samples
        paras['min_samples_leaf'] = leaf_samples

    elif  method == 'cal': 
        paras = {}
        min_samples_split,min_samples_leaf = get_min_samples(len(df))
        clf.min_samples_split = min_samples_split
        clf.min_samples_leaf = min_samples_leaf
        clf.fit(X,y)
        depth = tree_prune.get_max_depth(clf)
        leaf_samples,node_samples = tree_prune.get_min_sample_leaf_split(clf)       
        paras['max_depth'] = depth
        paras['min_samples_split'] = node_samples
        paras['min_samples_leaf'] = leaf_samples
        
    elif method == 'tune':
        paras = tune_paras(clf,X,y, 
                           random_state = random_state,
                           cv =cv, 
                           scoring = scoring,
                           n_jobs =n_jobs)
    elif method == 'prune':
        #clf_prune.max_depth = depth
        #clf_prune.min_samples_split = node_samples
        #clf_prune.min_samples_leaf = leaf_samples


        if depth >= 25:
            clf_prune.max_depth = 25
        else:
            clf_prune.max_depth = depth        
        
        min_samples_split,min_samples_leaf = get_min_samples(len(y))
        clf_prune.min_samples_split = int(0.8*min_samples_split)
        clf_prune.min_samples_leaf = int(0.8*min_samples_leaf)
        
        clf_prune.fit(X,y)
        
        clf1 = deepcopy(clf_prune)

        print('get optimal n_leaves,max leaves of the tree is %d......\n' % tree_prune.get_n_leaves(clf1))  
        
        n_leaves,means,stds,x = get_optimal_leaves(clf1,X,y,n_iterations=n_iter,random_state = random_state, alpha = alpha)
        
        print('optimal n_leaves is: %d, begin pruning tree...\n' % n_leaves)
        #prune
        clf_prune.prune(n_leaves)
        
        print('pruning is finished')
        
        #get pruned tree's best parameters
        max_depth = tree_prune.get_max_depth(clf_prune)
        min_sample_leaf, min_sample_split = tree_prune.get_min_sample_leaf_split(clf_prune)
        
        paras = {}
        paras['max_depth'] = max_depth
        paras['min_samples_split'] = min_sample_split
        paras['min_samples_leaf'] = min_sample_leaf      
        
        plot_prune(x,means,stds)
        
    elif method == 'both':
        
        print('tuning,please wait...\n')
        tune_dict = tune_paras(clf,X,y, 
                               random_state = random_state,
                               cv =cv, 
                               scoring = scoring,
                               n_jobs =n_jobs)
    
        clf_prune.max_depth = tune_dict['max_depth']
        clf_prune.min_samples_split = tune_dict['min_samples_split']
        clf_prune.min_samples_leaf = tune_dict['min_samples_leaf']
        
        
        clf_prune.fit(X,y)


        clf2= deepcopy(clf_prune)
        
        print('get optimal n_leaves...,max leaves of the tree is %d\n' % tree_prune.get_n_leaves(clf2))          
        
        #find best prune parameter: n_leaves
        n_leaves,means,stds,x = get_optimal_leaves(clf2,X,y,n_iterations=n_iter,random_state = random_state, alpha = 0.85)

        print('optimal n_leaves is: %d, begin pruning tree...\n' % n_leaves)
        
        #prune
        clf_prune.prune(n_leaves)
        
        print('pruning is finished') 
        
        #get pruned tree's best parameters
        max_depth = tree_prune.get_max_depth(clf_prune)
        min_sample_leaf, min_sample_split = tree_prune.get_min_sample_leaf_split(clf_prune)
        
        paras = {}
        paras['max_depth'] = max_depth
        paras['min_samples_split'] = min_sample_split
        paras['min_samples_leaf'] = min_sample_leaf    
        
        plot_prune(x,means,stds)
    else:
        print("get empty parameters dict, only \n 'none', 'cal', 'tune', 'prune' and 'both' \n are supported in the method currently")
        paras = {} 
        
    if method == 'both' or 'prune':
        return paras,clf_prune 
    else:
        return paras,clf
        


def tree_features(df, final_paras, seed_list = list(range(0,500,50))):
    '''
    input:
        X,y: array
        final_paras: dict of paras in tree, can be empty

    output: 
        feature_importance    
    '''
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    
    feature_names = df.columns[:-1]
    
    if len(np.unique(y)) >= 5:
        clf = tree.DecisionTreeRegressor()
    else:
        clf = tree.DecisionTreeClassifier()
    
    
    if 'max_depth' in final_paras.keys():
        clf.max_depth = final_paras['max_depth']
    if 'max_leaf_nodes' in final_paras.keys():
        clf.max_leaf_nodes = final_paras['max_leaf_nodes']
    if 'min_samples_leaf' in final_paras.keys():
        clf.min_samples_leaf = final_paras['min_samples_leaf']

    dfall =pd.DataFrame(index= feature_names)
      
    for i in seed_list:
        clf.random_state = i
        clf.fit(X,y)
        coef = pd.DataFrame(clf.feature_importances_,columns = [i],index= feature_names)
        dfall = dfall.join(coef)
    dfall.columns.name = 'random_seed'
    return dfall.mean(axis = 1).sort_values(ascending = False)



 
if __name__ == "__main__":
    #df = pd.read_csv('../data/new/prepared data/bairong_train.csv',index_col = 'uid')
    
    #df = pd.read_csv('../data/new/prepared data/DPFINAL_train.csv',index_col = 'Customer_ID')   
    
    df = pd.read_csv('../data/new/prepared data/CONT_train.csv',index_col = 'ID')   
    
    #train,test = train_test_split(df, test_size=0.3, random_state=42)
    
    df = df[df.columns[1:]]
    '''
    paras,clf = get_optimal_paras(train, 
                                  scoring = 'roc_auc', 
                                  method = 'both', 
                                  n_iter = 10, 
                                  cv = 5, 
                                  n_jobs = 1,
                                  random_state = 100)
    '''
    
    all_paras={}
    for method in ['none','cal','tune','prune','both']: 
        print('method of %s .....\n' % method)
        start = time()
        paras,clf = get_tree_paras(df, 
                              method = method, 
                              n_iter = 5, 
                              cv = 3, 
                              n_jobs = -1,
                              random_state = 100,
                              alpha = 0.5)
        end =time()
        interval = end - start
        paras['time_used(s)'] = interval
        all_paras[method]=paras
    #pd.DataFrame.from_dict(all_paras).to_csv('../result/bairong_paras.csv')
    #pd.DataFrame.from_dict(all_paras).to_csv('../result/dpfinal_paras.csv')
    pd.DataFrame.from_dict(all_paras).to_csv('../result/cont_paras.csv')
    

'''
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data = export_graphviz(clf, out_file=None,feature_names=df.columns[1:-1],  
                         filled=True, rounded=True,  
                         special_characters=True
                         )
graph = pydotplus.graphviz.graph_from_dot_data(dot_data)
Image(graph.create_png())
''' 
    
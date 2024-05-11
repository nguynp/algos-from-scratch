import numpy as np
import pandas as pd
import concurrent.futures
from collections import Counter
import random
from math import ceil

'''Reuse almost code from Decision Tree model'''
class Tree_Node():
    def __init__(self, feat_id = None, split_val = None,
    left = None, right = None, *,label = None):
                
        self.feat_id = feat_id # feature that the node include
        self.split_val = split_val # value that the node include
        
        self.right = right # right branch
        self.left = left # left branch

        self.label = label # class label

class Decision_Tree_Classification():

    def __init__(self, root=None, max_depth=None, min_split_val=None):
        self.root = root
        self.max_depth = max_depth
        self.min_split_val = min_split_val

    ''' find indices to split when threshold divide data into 2 parts '''

    def split_idx(self, X, feat_id, threshold):

        left_idx = np.argwhere(X[:, feat_id] == threshold).flatten()
        right_idx = np.argwhere(X[:, feat_id] != threshold).flatten()

        return (left_idx, right_idx)

    ''' compute information gain with entropy or gini splitting criteria '''

    def gini(self, y):  # compute gini index

        uni, uni_count = np.unique(y, return_counts=True)
        count = len(y)
        prob = uni_count / count
        gini_val = 1 - np.square(prob).sum()

        return (gini_val)

    def entropy(self, y):  # compute entropy
        uni, uni_count = np.unique(y, return_counts=True)
        count = len(y)
        prob = uni_count / count
        entropy = (-prob*np.log2(prob)).sum()

        return (entropy)

    # entropy splitting criterion as default
    def information_gain(self, X, y, feat_id, threshold, criterion='entropy'):

        left_idx, right_idx = self.split_idx(X, feat_id, threshold) 
        prob_left = len(left_idx)/len(y)
        prob_right = len(right_idx)/len(y)

        if criterion == 'entropy':
            inf_gain = self.entropy(y) - (self.entropy(y[left_idx])*prob_left + self.entropy(y[right_idx])*prob_right)
        else:
            inf_gain = self.gini(y) - (self.gini(y[left_idx])*prob_left + self.gini(y[right_idx])*prob_right)

        return (inf_gain)

    ''' find indices to split with highest information gain '''

    def best_split(self, X, y):

        best_gain = -100
        best_feat_id = None
        best_threshold = None

        # loop over each feature 
        for i in range(len(X.T)):

            for value in np.unique(X[:, i]):

                gain = self.information_gain(X, y, i, value)

                if gain > best_gain:
                    best_gain = gain
                    best_feat_id = i
                    best_threshold = value
                else:
                    continue

        return (best_feat_id, best_threshold)

    ''' build tree based on best splits and apply to unseen data '''

    # new added for bagging
    def random_feats_to_split(self, X):

        # import random

        t_feats = X.shape[1] # total num of features 
        n_r_feats = random.randint(1, t_feats) # get random num of feats for splitting
        rfeats_idx = sorted(random.sample(list(range(t_feats)), n_r_feats))

        X_r = X[:, rfeats_idx] # X with random feats

        return X_r, rfeats_idx

    def build_tree(self, X, y, depth=0):

        # stop splitting condition
        if (len(np.unique(y)) == 1) or (depth > self.max_depth) or (len(X) < self.min_split_val):
            # leaf node with class label
            return Tree_Node(label=self.majority_label(y))

        else:  # continue to split

            # randomly select features
            X_r, rfeats_idx = self.random_feats_to_split(X)
            feat_id_X_r, value = self.best_split(X_r, y) # find best split based on X with random feats
            feat_id = rfeats_idx[feat_id_X_r] # refer back to right feat id of X

            # use best split feat&val to split on original X
            left_idx, right_idx = self.split_idx(X, feat_id, value) 

            # prevent splitting with empty value in 1 branch
            if len(left_idx) == 0 or len(right_idx) == 0:
                # leaf node with class label
                return Tree_Node(label=self.majority_label(y))
            else:  # new sub tree
                left = self.build_tree(X[left_idx, :], y[left_idx], depth+1)
                right = self.build_tree(X[right_idx, :], y[right_idx], depth+1)
                return Tree_Node(feat_id, value, left, right)

    def majority_label(self, y):  # find most common label
        unique_classes, counts_unique_classes = np.unique(
            y, return_counts=True)
        id = counts_unique_classes.argmax()
        return unique_classes[id]

    def fit(self, X, y):  # fit the decision tree on dataset
        self.root = self.build_tree(X, y)

    # reverse the tree to predict a new test instance label
    def predict_inst(self, x_test, tree=None):

        if not tree:
            tree = self.root

        if x_test[tree.feat_id] == tree.split_val:
            node = tree.left
        else:
            node = tree.right

        if node.label != None:  # if the node is a leaf with class label
            return node.label

        if node.label == None:  # if the node is a sub tree
            sub_tree = node
            return self.predict_inst(x_test, sub_tree)

    def predict_all(self, Xtest):  # predict all values in test set
        pred = Xtest.apply(lambda x: self.predict_inst(x), axis=1)
        return pred
    
class Random_Forest_Classification():
    ''' 
    For: multi-categorical-value prediction
    
    Input of df of every function is a Dateframe format,
        class values are in the last column of df
    '''
    def __init__(self, forest_= None, list_sampling_idx = None,
                *, ntree = None, max_depth_= None, min_split_val_= None):

        self.forest_ = forest_ # list of trees
        self.ntree = ntree # number of trees

        self.max_depth_ = max_depth_ 
        self.min_split_val_ = min_split_val_ 

        self.list_sampling_idx = list_sampling_idx # (for defining oob)

    '''    randomization part: bootstrap input data
    '''
    def create_bootstraped_df(self, df_ori):
        df_boot = df_ori.sample(len(df_ori), replace=True)
        return df_boot

    '''    build forest based on train set and apply to unseen data
    '''
    def build_tree(self, df_ori):

        # randomize
        df_train = self.create_bootstraped_df(df_ori)
        # df_train = self.select_random_features(df_boot)

        # get unique sampling rows in bootstraped df (for defining oob)
        row_idx = list(df_train.index.unique())

        # prepare input (array format for decision tree)
        y_train = np.array(df_train.iloc[:,-1])
        X_train = np.array(df_train.iloc[:,:-1])

        # build a tree
        tree_classify = Decision_Tree_Classification(max_depth = self.max_depth_, min_split_val = self.min_split_val_)
        tree_classify.fit(X_train, y_train)

        return tree_classify, row_idx
    
    def build_forest(self, df_ori):
        # import concurrent.futures
        forest = []

        list_sampling_idx = [] 

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(self.build_tree, df_ori) for _ in range(self.ntree)]
        for f in concurrent.futures.as_completed(results):
            tree, sampling_idx = f.result()
            forest.append(tree) # a forest with n trees
            list_sampling_idx.append(sampling_idx) # for defining oob

        return forest, list_sampling_idx 


    '''    main functions to call: fit & predict
    '''
    def fit(self, df_ori, n_tree): # fit the random forest on input dataset
        self.ntree = n_tree
        self.forest_, self.list_sampling_idx = self.build_forest(df_ori)

    def predict(self, df_test):

        '''df_test: Dataframe format, contains both features & class column'''
        # import concurrent.futures
        # from collections import Counter
        n_tree = self.ntree
        list_preds = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            preds = [
                executor.submit(self.forest_[i].predict_all,df_test.iloc[:,:-1]) 
                                    for i in range(n_tree)
                                    ]
        for f in concurrent.futures.as_completed(preds):
            list_preds.append(f.result()) # each pred is a series of predicted class
       
        # get all preds value from n trees to one dataframe for ensambling   
        all_pred_forest = pd.DataFrame({str(i+1): preds for i, preds in enumerate(list_preds)})
        agg_preds = all_pred_forest.apply(lambda x: Counter(x).most_common(1)[0][0], axis=1)  

        return agg_preds # series of predicted class after aggregrating all trees in the forest

    '''    evaluate Random Forest model with out-of-bag (oob) error metric
    '''

    def find_trees_of_oob_sample(self, row_id):
        n_tree = self.ntree
        tree_idx = [] # list of trees idx which that sampling point is OOB
        for i in range(n_tree):
            if row_id not in self.list_sampling_idx[i]:
                tree_idx.append(i)

        return tree_idx # trees using for predicting that oob value

    def predict_oob_sample(self, row, row_id):    
        tree_idx = self.find_trees_of_oob_sample(row_id)
        x_test = np.array(row[:-1]) # remove class -> input for decision tree
        preds = []
        for i in tree_idx:
            pred = self.forest_[i].predict_inst(x_test)
            preds.append(pred)
        if len(preds) > 0: # check if that point is an oob 
            label = Counter(preds).most_common(1)[0][0] # find most common class
            return label
        else: 
            return np.nan

    def predict_all_oob_points(self, df_ori):
        df_pred_oob = df_ori.reset_index() # get id of all rows
        df_pred_oob['pred_oob'] = df_pred_oob.apply(lambda x: self.predict_oob_sample(x[1:], x['index']), axis = 1)
        
        # filter out none oob values
        df_pred_oob = df_pred_oob[~df_pred_oob['pred_oob'].isna()]

        # df_pred_oob.drop(columns = 'index', axis=0)
        return df_pred_oob['pred_oob']
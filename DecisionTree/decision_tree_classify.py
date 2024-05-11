import numpy as np
import pandas as pd


class Tree_Node():

    def __init__(self, feat_id = None, split_val = None, feat_type = None, left = None, right = None, *,label = None):
                
        self.feat_id = feat_id # feature that the node include
        self.feat_type = feat_type # numerical or categorical
        self.split_val = split_val # value that the node include
        
        self.right = right # right branch
        self.left = left # left branch

        self.label = label # class label

class Decision_Tree_Classification():

    def __init__(self, root=None, max_depth=None, min_split_val=None, list_feat = None):
        self.root = root
        self.max_depth = max_depth
        self.min_split_val = min_split_val

        self.list_feat = list_feat 

    ''' define whether every feature is numerical or categorical '''

    def define_feature_types(self, X, thres_condition=100):

        f_type = []
        for i in range(len(X.T)):
            # determined by number of unique values of every feature
            if len(np.unique(X[:, i])) > thres_condition:
                f_type.append('num')
            else:
                f_type.append('categor')

        return f_type  # list of features types

    ''' find indices to split when threshold divide data into 2 parts '''

    def split_idx(self, X, feat_id, threshold, f_type):

        if f_type == 'num':  # when feature is numerical
            left_idx = np.argwhere(X[:, feat_id] <= threshold).flatten()
            right_idx = np.argwhere(X[:, feat_id] > threshold).flatten()

        else:  # when feature is categorical
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
    def information_gain(self, X, y, feat_id, threshold, f_type, criterion='entropy'):

        left_idx, right_idx = self.split_idx(X, feat_id, threshold, f_type)

        prob_left = len(left_idx)/len(y)
        prob_right = len(right_idx)/len(y)

        if criterion == 'entropy':
            inf_gain = self.entropy(
                y) - (self.entropy(y[left_idx])*prob_left + self.entropy(y[right_idx])*prob_right)
        else:
            inf_gain = self.gini(
                y) - (self.gini(y[left_idx])*prob_left + self.gini(y[right_idx])*prob_right)

        return (inf_gain)

    ''' find indices to split with highest information gain '''

    def best_split(self, X, y, step=200):

        best_gain = -100
        best_feat_id = None
        best_threshold = None
        best_ftype = None
        f_type_list = self.define_feature_types(X)

        # loop over each feature (and its type)
        for i, ftype in zip(range(len(X.T)), f_type_list):

            if ftype == 'num':  # loop over range of numerical value with defined step
                min = np.min(X[:, i])
                max = np.max(X[:, i])
                for value in np.arange(min, max, ((max-min)/step)):

                    gain = self.information_gain(X, y, i, value, 'num')

                    if gain > best_gain:
                        best_gain = gain
                        best_feat_id = i
                        best_threshold = value
                        best_ftype = ftype
                    else:
                        continue

            else:  # loop over each unique categorical value
                for value in np.unique(X[:, i]):

                    gain = self.information_gain(X, y, i, value, 'categor')

                    if gain > best_gain:
                        best_gain = gain
                        best_feat_id = i
                        best_threshold = value
                        best_ftype = ftype
                    else:
                        continue

        return (best_feat_id, best_threshold, best_ftype)

    ''' build tree based on best splits and apply to unseen data '''

    def build_tree(self, X, y, depth=0):

        # stop splitting condition
        if (len(np.unique(y)) == 1) or (depth > self.max_depth) or (len(X) < self.min_split_val):
            # leaf node with class label
            return Tree_Node(label=self.majority_label(y))

        else:  # continue to split

            feat_id, value, feat_type = self.best_split(X, y)
            left_idx, right_idx = self.split_idx(X, feat_id, value, feat_type)

            # prevent splitting with empty value in 1 branch
            if len(left_idx) == 0 or len(right_idx) == 0:
                # leaf node with class label
                return Tree_Node(label=self.majority_label(y))
            else:  # new sub tree
                left = self.build_tree(X[left_idx, :], y[left_idx], depth+1)
                right = self.build_tree(X[right_idx, :], y[right_idx], depth+1)
                return Tree_Node(feat_id, value, feat_type, left, right)

    def majority_label(self, y):  # find most common label
        unique_classes, counts_unique_classes = np.unique(
            y, return_counts=True)
        id = counts_unique_classes.argmax()
        return unique_classes[id]

    def fit(self, X, y):  # fit the decision tree on dataset
        self.root = self.build_tree(X, y)

    # reverse the tree to predict a new test instance label
    def classify_inst(self, x_test, tree=None):

        if not tree:
            tree = self.root
        
        if tree.feat_type == 'num':  # numerical value
            if x_test[tree.feat_id] <= tree.split_val:
                node = tree.left
            else:
                node = tree.right
        else:  # categorical value
            if x_test[tree.feat_id] == tree.split_val:
                node = tree.left
            else:
                node = tree.right

        if node.label != None:  # if the node is a leaf with class label
            return node.label

        if node.label == None:  # if the node is a sub tree
            sub_tree= node
            return self.classify_inst(x_test, sub_tree)

    def class_prediction(self, Xtest):  # predict all values in test set
        pred = []
        for row in Xtest:
            p = self.classify_inst(row, self.root)
            pred.append(p)
        return np.array(pred)


    def print_tree(self, df = None, tree=None, indent=""):
        
        if not tree:
            tree = self.root
        if df is not None:
            self.list_feat = df.columns

        if tree.label is not None:
            print(tree.label)

        else:
            print("\n"+ indent + "⊛ "+str(self.list_feat[tree.feat_id]), "=", tree.split_val, "?")
            print("%s-> left (true): " % (indent), end="")
            self.print_tree(None, tree.left, "   " + indent)
            print("%s-> right (false): " % (indent), end="")
            self.print_tree(None, tree.right, "   " + indent)


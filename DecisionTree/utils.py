import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from decision_tree_classify import *

def train_test_split(df, test_size, class_name_col):
    df_test = df.sample(frac = test_size, random_state=100)
    df_train = df.drop(df_test.index)

    y_train = np.array(df_train.iloc[:,-1])
    X_train = np.array(df_train.drop([class_name_col], axis=1))

    y_test = np.array(df_test.iloc[:,-1])
    X_test = np.array(df_test.drop([class_name_col], axis=1))

    return y_train, X_train, y_test, X_test

def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct)/len(y_true)



# cross validation function with 5 fold
def DT_cross_validation(X, y, depth, min_split):
    
    kf = StratifiedKFold(n_splits=5) #from sklearn.model_selection import KFold

    accuracy_list = []
    tp_list = []
    tn_list = []
    fp_list = []
    fn_list = []
    for train, test in kf.split(X, y):
        
        train_id = list(train)
        test_id = list(test)

        X_f_train = X[train_id,:]
        y_f_train = y[train_id]
        X_f_test = X[test_id,:]
        y_f_test = y[test_id]

        model = Decision_Tree_Classification(max_depth = depth, min_split_val=min_split)

        model.fit(X_f_train, y_f_train)

        y_f_pred = model.class_prediction(X_f_test)


        # evaluation metrics
        acc = accuracy(y_f_pred, y_f_test)       
        accuracy_list.append(acc)


        tp = np.sum(np.logical_and(y_f_test == 1, y_f_pred == 1))
        tn = np.sum(np.logical_and(y_f_pred == 0, y_f_test == 0))
        fp = np.sum(np.logical_and(y_f_pred == 1, y_f_test == 0))
        fn = np.sum(np.logical_and(y_f_pred == 0, y_f_test == 1))

        # pred_list.append(y_)
        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
    # Precision
    # p = tp_list/(tp_list+fp_list)
    p = [tp_list[i]/ (tp_list[i]+fp_list[i]) for i in range(len(tp_list))]

    # Recall
    r = [tp_list[i]/ (tp_list[i]+fn_list[i]) for i in range(len(tp_list))]

    # F-score
    # F = (2*p*r)/(p+r)
    F = [(2*p[i]*r[i])/(p[i]+r[i]) for i in range(len(tp_list))]
    
    # return (accuracy_list)

    return(pd.DataFrame({"Accuracy": accuracy_list, "True Positive": tp_list, "True Negative": tn_list, "False Positive": fp_list, "False Negative": fn_list, "Precision": p, "Recall": r, "F-score": F}))#, tp_list)
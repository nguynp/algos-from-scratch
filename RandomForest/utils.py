import numpy as np

def train_test_split(df, test_size, class_name_col):
    df_test = df.sample(frac = test_size, random_state=1)
    df_train = df.drop(df_test.index)
    return df_test.reset_index(drop=True), df_train.reset_index(drop=True)

def accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    return float(correct)/len(y_true)
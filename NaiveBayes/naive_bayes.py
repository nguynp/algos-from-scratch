import pandas as pd
import numpy as np

# For categorical data
class MultinomialNaiveBayes:
    def __init__(self):
        self.prob_tables = {}
        self.class_log_probs = None
        self.class_col = None
        self.smoothing_value = 1e-9  # small probability for unseen features

    def fit(self, x_train, y_train):
        self.class_col = y_train.name
        df = pd.concat([x_train, y_train], axis=1)

        # Log prior of classes
        class_probs = y_train.value_counts(normalize=True)
        self.class_log_probs = np.log(class_probs)

        # Conditional log probabilities for each feature given class
        for feature in x_train.columns:
            # Counts for each (class, feature_value)
            counts = df.groupby([self.class_col, feature]).size()

            # Counts for each class
            class_counts = df.groupby(self.class_col).size()

            # Conditional probabilities P(feature_value | class)
            cond_probs = counts / class_counts.loc[counts.index.get_level_values(0)].values

            # Store log probabilities for fast multiplication as sum
            self.prob_tables[feature] = np.log(cond_probs)

    def predict_inst(self, x):
        log_probs = {}

        for cls in self.class_log_probs.index:
            log_p = self.class_log_probs[cls]

            for feature in x.index:
                
                if feature not in self.prob_tables: # skip feature if not seen in training
                    continue

                prob_table = self.prob_tables[feature]
                val = x[feature]

                try:
                    log_p += prob_table.loc[(cls, val)]
                except KeyError:
                    log_p += np.log(self.smoothing_value) #  → small smoothing value for unseen value

            log_probs[cls] = log_p

        max_log = max(log_probs.values())
        exps = {cls: np.exp(log_p - max_log) for cls, log_p in log_probs.items()}
        total = sum(exps.values())
        probs = {cls: val / total for cls, val in exps.items()}

        return pd.Series(probs)


    def predict_proba(self, X):
        return X.apply(self.predict_inst, axis=1)

    def predict(self, X):
        return self.predict_proba(X).idxmax(axis=1)

# For continuous data
class GaussianNaiveBayes:
    def __init__(self):
        self.class_stats = None
        self.class_probs = None
        self.class_col = None

    def fit(self, x_train, y_train):
        self.class_col = y_train.name
        df = pd.concat([x_train, y_train], axis=1)

        # compute mean and std of each feature by class/ target prediction
        self.class_stats = df.groupby(self.class_col).agg(['mean', 'std'])

        # compute class/ target prediction probability
        self.class_probs = y_train.value_counts(normalize=True)

    @staticmethod
    def normal_pdf(x, mu, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mu) / std) ** 2)

    def predict_inst(self, x):
        probs = {}
        for cls in self.class_probs.index:
            p = self.class_probs[cls]
            for feature in x.index:
                mu = self.class_stats.loc[cls, (feature, 'mean')]
                std = self.class_stats.loc[cls, (feature, 'std')]
                p *= self.normal_pdf(x[feature], mu, std)
            probs[cls] = p

        total = sum(probs.values())
        return pd.Series({cls: p / total for cls, p in probs.items()})

    def predict_proba(self, X):
        return X.apply(self.predict_inst, axis=1)

    def predict(self, X):
        return self.predict_proba(X).idxmax(axis=1)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from FIR_model import Model
from FIR_functions import add_noise

import numpy as np

# extract the data
dataset = load_breast_cancer()

features = dataset['feature_names']
# 10 features might make more sense, 11-30 are error and worst values won't really work with noise
features = features[:10]

X = dataset['data']
X = np.delete(X, slice(10, 30, 1), 1)
y = dataset['target']

X = add_noise(X, features, 7)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
n_test_samples = len(X_test)

# use the model
model = Model(10, 15, 12, n_test_samples)
model.evaluate(X_train, y_train, X_test, y_test)
model.rank(X_test, y_test, features)

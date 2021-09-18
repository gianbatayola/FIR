from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from FIR_model import Model
from itertools import combinations
import time
import pandas as pd
import numpy as np

df = pd.read_csv('heart.csv')

features = df.columns[:13]

X = df.values[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
# scale = StandardScaler()
# scale.fit(X)
X = StandardScaler().fit_transform(X)
y = df.values[:, 13]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
n_test_samples = len(X_test)

model = Model(13, 8, 8, n_test_samples)
model.evaluate(X_train, y_train, X_test, y_test)
rank = model.rank(X_train, y_train, features)
print(rank)


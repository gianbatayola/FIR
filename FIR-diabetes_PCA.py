import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('diabetic_data.csv')
df=df.replace('?',np.nan).dropna(axis = 0, how = 'any') # for sake of ease just remove anything with a ? 

# turn our target variable into a binary
df.loc[df['readmitted'] == 'NO', 'readmitted'] = 0
df.loc[df['readmitted'] == '>30', 'readmitted'] = 1
df.loc[df['readmitted'] == '<30', 'readmitted'] = 1

df['readmitted'] = df['readmitted'].apply(np.int64) # make sure we got int

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df = df.select_dtypes(include = numerics) # only keep numerics, find better way

X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values
y = df.iloc[:, 13].values

model = PCA().fit(X)

X_pc = model.transform(X)

n_pcs= model.components_.shape[0]

most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = df.columns[:13]

most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

df2 = pd.DataFrame(dic.items())

print(df2)

scale = StandardScaler()
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

print(df.columns)

# top 3 features
X_train_3 = X_train[:,[0, 1, 6]] 
X_test_3 = X_test[:, [0, 1, 6]]
clf = MLPClassifier((10, 10), random_state=1)
clf.fit(X_train_3, y_train)
print(clf.score(X_test_3, y_test))

# top 6 features
X_train_6 = X_train[:,[0, 1, 3, 4, 6, 8]] 
X_test_6 = X_test[:,[0, 1, 3, 4, 6, 8]] 
clf = MLPClassifier((10, 10), random_state=1)
clf.fit(X_train_6, y_train)
print(clf.score(X_test_6, y_test))

# top 9 features
X_train_9 = X_train[:,[0, 1, 3, 4, 5, 6, 7, 8, 9]] 
X_test_9 = X_test[:, [0, 1, 3, 4, 5, 6 ,7 ,8 ,9]]
clf = MLPClassifier((10, 10), random_state=1)
clf.fit(X_train_9, y_train)
print(clf.score(X_test_9, y_test))

# top 12 features
#X_train_12 = X_train[:,[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]] 
#X_test_12 = X_test[:, [0, 1, 3, 4, 5, 6 ,7 ,8 ,9, 10, 11]]
#clf = MLPClassifier((10, 10), random_state=1)
#clf.fit(X_train_12, y_train)
#print(clf.score(X_test_12, y_test))

# with all features
clf = MLPClassifier((10, 10), random_state=1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('dataset_diabetes/diabetic_data.csv')
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

df = pd.DataFrame(dic.items())

print(df)

plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


scale = StandardScaler()
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
n_test_samples = len(X_test)

clf = MLPClassifier((10, 10), random_state=1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test)) #we threw out everything important lolol







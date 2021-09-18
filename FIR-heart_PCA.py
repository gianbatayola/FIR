import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.decomposition import PCA

dataset = pd.read_csv('heart.csv')

X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values
y = dataset.iloc[:, 13].values

model = PCA().fit(X)

X_pc = model.transform(X)

n_pcs= model.components_.shape[0]

most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = dataset.columns[:13]

most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

df = pd.DataFrame(dic.items())

print(df)

# X = StandardScaler().fit_transform(X)
#
# pca = PCA()
#
# pca.fit_transform(X)
#

#print(pca.explained_variance_)
#print(pca.n_components_)

#pdf = pd.DataFrame(data=pc, columns=feat_names)

#fdf = pd.concat([pdf, dataset['target']], axis=1)

#print(fdf)

# plt.title('2 component PCA')
# plt.xlabel('Principal component 1')
# plt.ylabel('Principal component 2')
# plt.scatter(fdf['principal component 1'], fdf['principal component 2'], c=fdf['target'])
# plt.savefig('pca.png')
#
# plt.show()

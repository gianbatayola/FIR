from sklearn.datasets import load_breast_cancer
import pandas as pd

dataset = load_breast_cancer()

features = dataset['feature_names']

#print(features)

datasett = pd.read_csv('heart.csv')

print(datasett.columns)

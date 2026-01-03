import numpy as np
import pandas as pd

train = pd.read_csv("http://bit.ly/fc-ml-titanic")
print(train.head())

features = ['Pclass', 'Sex', 'Age', 'Fare']
label = ['Survived']

print(train[features].head())
print(train[label].head())
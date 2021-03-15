# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 23:48:46 2021

@author: orkun.yuzbasioglu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from knn import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn import neighbors

def accuracy(y_true, y_pred):
    
    matches = len([i for i,j in zip(y_true, y_pred) if i == j])
    samples = len(y_true)
    
    try:
        return matches/samples
    except ZeroDvisionError:
        return 0
    
def error_count(y_true, y_pred):
    errors = len([i for i,j in zip(y_true, y_pred) if i != j])
    samples = len(y_true)
    
    return '{}/{}'.format(errors, samples)

rows = []

with open('iris_data.txt', 'r') as f:
    
    for line in f:
        
        line_fields = line.rstrip().split(',') 
        rows.append([float(line_fields[0]), float(line_fields[3]), line_fields[4]])

X_all = []
y_all = []
X_train = []
y_train = []
X_test = []
y_test = []

current_class = rows[0][2]
counter = 1

for record in rows:
    
    X_all.append([record[0],record[1]])
    y_all.append(record[2])
    new_class = record[2] 
    
    if new_class != current_class:
        counter = 1
        current_class = new_class
        
    if counter <= 30:
        X_train.append([record[0],record[1]])
        y_train.append(record[2])
    else:
        X_test.append([record[0],record[1]])
        y_test.append(record[2])
        
    counter = counter + 1
    
#with class
print("K-NN implemenation:")
for k in range(1,17,2):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    print('For k = {}, Accuracy = {:.2f}, Error Count = {}'.format(k, accuracy(y_test, y_pred), error_count(y_test, y_pred)))
    
#sklearn's implementation
print("\nSklearn's K-NN implemenation:")
for k in range(1,17,2):
    neigh = neighbors.KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    
    print('For k = {}, Accuracy = {:.2f}, Error Count = {}'.format(k, accuracy(y_test, y_pred), error_count(y_test, y_pred)))
    
X = np.array(X_train)
y = pd.factorize(y_train)[0]

target_names =  np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
n_neighbors = 1

# step size in the mesh
h = .02  

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']

#  create an instance of Neighbours Classifier and fit the data.
# clf = neighbors.KNeighborsClassifier(n_neighbors)
# clf.fit(X, y)
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = np.array(clf.predict(np.c_[xx.ravel(), yy.ravel()]))

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=target_names[y], palette=cmap_bold, alpha=1.0, edgecolor="black")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % (n_neighbors))
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[3])

plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 23:49:18 2021

@author: orkun.yuzbasioglu
"""

from Point import Point
from collections import Counter

class KNeighborsClassifier:
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.training_data = []
        
    def fit(self, X, y):
        self.training_data = [Point(el[0][0],el[0][1],el[1]) for el in zip(X,y)]
        
    def predict(self, X):
        
        test_data = [Point(el[0], el[1]) for el in X]
        predicted_labels = []
        
        for test_point in test_data:
            
            distances=[]
            
            for training_point in self.training_data:
                
                distances.append(training_point.distance(test_point))
    
            indexes = sorted(range(len(distances)), key=lambda k: distances[k])[:self.n_neighbors]
            neighbours_counter = Counter([self.training_data[i].label for i in indexes])
            neighbours_dict = dict(neighbours_counter) 
            predicted_label = max(neighbours_dict, key=neighbours_dict.get)
            predicted_labels.append(predicted_label)
            
        return(predicted_labels)

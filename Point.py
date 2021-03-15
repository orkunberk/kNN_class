# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 23:49:04 2021

@author: orkun.yuzbasioglu
"""

class Point:
    
    def __init__(self, x, y, label=''):
        self.x = x
        self.y = y
        self.label = label
        
    def get_coordinates(self):
        return [self.x, self.y]
    
    def get_label(self):
        return self.label
    
    def __str__(self):
        return 'X position: {}, Y position: {}, label: {}'.format(self.x, self.y, self.label)
        
    def distance(self, other_point):
        dx = self.x - other_point.x
        dy = self.y - other_point.y
        distance = ((dx)**2 + (dy)**2)**0.5
        return distance
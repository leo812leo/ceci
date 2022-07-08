# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:40:12 2022

@author: 63791
"""
import numpy as np
from numpy import matrix
from numpy.linalg import norm



def stress_transform(stress_matrix, local_axis):
    
    def unit_vector(vector):
        return vector / norm(vector)
    
    transform_matrix = []
    for vec in local_axis:
        transform_matrix.append(unit_vector(vec))
    T = matrix(transform_matrix)
    stress_matrix = matrix( stress_matrix )
    return  T @ stress_matrix @ T.T


def extrapolation(data_a, data_b, toe):
    
    def dist(p1,p2):
        return np.sqrt(np.sum(np.square(p1-p2)))
    
    a, sigma_a = data_a
    b, sigma_b = data_b
    d1 = dist(a,b)
    d2 = dist(a,toe)
     
    return  sigma_b + (sigma_a - sigma_b) * (d1 + d2) / d1
    
#ex1
   
stress = [[-50,-20, 0],
          [-20,-30, 0],
          [  0,  0, 0]]


stress_transform(stress,[[1,np.tan(31.7*np.pi/180),0],[-np.tan(31.7*np.pi/180),1,0],[0,0,1]])
stress_transform(stress,[[1,-0.2364,0],[np.tan(13.3*np.pi/180),1,0],[0,0,1]])


# #ex2

# stress = [[-120, 30, 0],
#           [  30,-60, 0],
#           [   0,  0, 0]]
# stress = np.matrix(stress)

# stress_transform(stress,[1,np.tan(-50*np.pi/180),0])
# stress_transform(stress,[1,-0.2364,0])
    



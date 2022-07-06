# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:40:12 2022

@author: 63791
"""
import numpy as np
from numpy import norm, cross, dot, array, matrix
from scipy.spatial.transform import Rotation as R



def stress_transform(stress_matrix,new_Xasis):
    
    def unit_vector(vector):
        return vector / norm(vector)
    
    def cal_axis_angle(vector): 
        vector = unit_vector( array(vector) )
        x = array([1,0,0])
        axis = unit_vector( cross (vector, x) )
        angle = np.arccos( dot(vector, x) )
        return axis, angle
    
    stress_matrix = matrix( stress_matrix )
    axis, theta = cal_axis_angle( new_Xasis )
    rot = R.from_rotvec(theta * axis)
    transform = rot.as_matrix()
    
    return ( transform @ stress_matrix @ transform.T ).round(3)
    
def extrapolation(data_a, data_b, toe):
    
    def dist(p1,p2):
        return np.sqrt(np.sum(np.square(p1-p2)))
    
    a, sigma_a = data_a
    b, sigma_b = data_b
    d1 = dist(a,b)
    d2 = dist(a,toe)
     
    return  sigma_b + (sigma_a - sigma_b) * (d1 + d2) / d1
    
# #ex1
   
# stress = [[-50,-20, 0],
#           [-20,-30, 0],
#           [  0,  0, 0]]
# stress = np.matrix(stress)


# stress_transform(stress,[1,0.6176,0])
# stress_transform(stress,[1,-0.2364,0])



# #ex2

# stress = [[-120, 30, 0],
#           [  30,-60, 0],
#           [   0,  0, 0]]
# stress = np.matrix(stress)

# stress_transform(stress,[1,np.tan(-50*np.pi/180),0])
# stress_transform(stress,[1,-0.2364,0])
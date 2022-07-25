# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:36:36 2021

@author: 63791
"""
import numpy as np
import copy
from rotation import stress_transform

def Newton(f,diff, ini_val = 1, err=10**-13, maxiter = 100, prt_step = False):
    x = ini_val 
    for i in range(maxiter):
        x -= f(x)/diff(x)    
        if prt_step == True:
            print(f"After {i+1} iteration, the solution is updated to {x}, f(x) is {f(x)}")
        if abs(f(x)/diff(x))<err:
            break
    return x

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def num(n):
    arr = []
    for i in range(n+1):
        arr.append(i)
    b = copy.deepcopy(arr)
    b.pop(-1)
    arr.reverse()
    return b + arr

def sign_convention(data,cb,in_out):
        U,U_IN,U_IF = data
        
        if in_out == '內側' and cb == '弦桿':
            return np.sign( np.dot(U_IN, np.cross(U, U_IF)) ),np.sign( np.dot(U_IN,U))
        elif in_out == '外側' and cb == '弦桿':
            return np.sign( np.dot(U_IN, np.cross(U, U_IF)) ),np.sign( np.dot(U_IN,U))            
        elif cb == '斜撐':           
            return np.sign( np.dot(U_IN, np.cross(U, U_IF)) ),np.sign( np.dot(U_IN,U))    

def dist(series):
    return np.sqrt(np.sum(series["x"]**2 + series["y"]**2 + series["z"]**2 ))

def local_stress_cal(series,local_axis):
    
    def array2mat(vec):
        mat = \
            [[vec[0], vec[3], vec[5]],
             [vec[3], vec[1], vec[4]],
             [vec[5], vec[4], vec[2]]]
        return np.array(mat)
    
    def mat2array(mat):
        vec = [mat[0,0], mat[1,1], mat[2,2], mat[0,1], mat[1,2], mat[0,2]]
        return np.array(vec)
    
    stresses = series.apply(array2mat)
    local_vec = list(local_axis[series.name].values())  #series.name = rad 每一列同一個local_axis
    return np.array([ mat2array(stress_transform(stress, local_vec))
                     for stress in stresses]).T

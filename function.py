# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:36:36 2021

@author: 63791
"""
import numpy as np

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
    word=''
    for i in range(n+1):
        word += str(i)  
    word += word[(n-1)::-1]
    return word

def sign_convention(data,cb,in_out):
        U,U_IN,U_IF = data
        
        if in_out == '內側' and cb == '弦桿':
            return np.sign( np.dot(U_IN, np.cross(U, U_IF)) ),np.sign( np.dot(U_IN,U))
        elif in_out == '外側' and cb == '弦桿':
            return np.sign( np.dot(U_IN, np.cross(U, U_IF)) ),np.sign( np.dot(U_IN,U))            
        elif cb == '斜撐':           
            return np.sign( np.dot(U_IN, np.cross(U, U_IF)) ),np.sign( np.dot(U_IN,U))    


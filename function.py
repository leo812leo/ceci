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

def Newton2d (point_j,point_k, ini_val = [0,3/2*np.pi], err=10**-13, maxiter = 100, prt_step = False):
    x1,x2 = ini_val
    for i in range(maxiter):
        j,k = point_j.inter(x1), point_k.inter(x2)
        diff = j[0]-k[0]
        diff_1 = 2 * np.array([ np.dot(j[1],diff) ,  np.dot(-k[1],diff) ])
        diff_2 = ( 2 * np.array( [               
                 np.dot(j[2],diff) + np.dot(j[1],j[1]), -np.dot(j[1],k[1]),
                -np.dot(j[1],k[1]), np.dot(k[2],-diff) + np.dot(k[1],k[1]) 
                ]) ).reshape((2,2))  
        diff_2inv = np.linalg.inv(diff_2)
        
        
        delta_x = np.matmul(diff_2inv, diff_1)
        x1, x2 = np.array([x1, x2]) - delta_x
        if prt_step == True:
            print(f"After {i+1} iteration,\nthe solution is updated to {round(x1,5),round(x2,5)}, delta_x is {np.linalg.norm(delta_x)}\n")
        if np.linalg.norm(delta_x)<err:
            break
    return x1, x2, np.linalg.norm(diff)
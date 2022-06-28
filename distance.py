# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:53:59 2021

@author: 63791
"""
import numpy as np
from function import unit_vector,Newton2d



class Joints():
# =============================================================================
# 屬性(Attribute)
# =============================================================================
    def __init__(self, chord, brace):
        self.chord = chord  # chord座標
        self.brace = brace  # brace座標 
        

    def printall(self):
        print(f"My chord is {chord}")    
        print(f"My brace is {brace}")
    
        
    def inter(self,theta):
        a,b,R,T = self.chord.values()
        c,d,r,t = self.brace.values()
        a,b,c,d = np.array([a,b,c,d])
        
        #座標單位向量
        Uc = unit_vector(b-a)                               #chord 中心線
        U_gamma = unit_vector(d-c)                          #brace 中心線
        U_alpha = unit_vector( np.cross(Uc, U_gamma) )      #chord 中心線 /brace 中心線 公垂線
        U_beta = unit_vector( np.cross(U_gamma,U_alpha) )   # (brace 平面)
    
        beta = np.dot(Uc,U_beta)
        gamma = np.dot(Uc,U_gamma)
        #0階
        C1 = np.dot(c-a , np.cross(U_alpha,Uc)) - r*gamma*np.sin(theta)
        C2 = R**2 - ( np.dot(c-a,U_alpha) + r*np.cos(theta))**2
        l = 1/beta *(-C1 + C2**0.5 )
        PI = c + r*np.cos(theta) * U_alpha + r*np.sin(theta) * U_beta  + l * U_gamma
        
        #1階
        C1_1 = -r*gamma*np.cos(theta)
        C2_1 = 2*r*np.sin(theta) * np.dot(c-a ,U_alpha) + 2* r**2 * np.cos(theta) * np.sin(theta)
        l_1 =  1/beta *( -C1_1+C2_1/(2*C2**0.5) )
        PI_1 = -r*np.sin(theta) * U_alpha + r*np.cos(theta) * U_beta  + l_1 * U_gamma
        
        #2階
        C1_2 = r*gamma*np.sin(theta)
        C2_2 = 2*r*np.cos(theta) * np.dot(c-a ,U_alpha) + 2* r**2 * ( np.cos(theta)**2 - np.sin(theta)**2 )
        l_2 =  1/beta * (-C1_2 + 0.5 * C2_2 * C2**-0.5 - 0.25 * C2_1**2 * C2**-1.5)
        PI_2 = -r*np.cos(theta) * U_alpha - r*np.sin(theta) * U_beta  + l_2 * U_gamma
        return [PI,PI_1,PI_2]
    
a = [-1421.00, 16246.00, -40360.00]
b = [-1421.00, 16246.00,-29660.00]
R = 1420
T = 105
# 斜撐中心線1
c1 = [-2064.00, 14980.00, -34128.00]
d1 = [-3709.00, 11079.00, -30614.00]
r1 = 717
t1 = 32

# 斜撐中心線2
c2 = [-568.00, 15111.00, -34078.00]
d2 = [1729.00, 11555.00, -30564.00]
r2 = 717
t2 = 32


# 其他
chord = {'a':a,'b':b,'R':R,'T':T}
brace1 = {'c':c1,'d':d1,'r':r1,'t':t1}
brace2 = {'c':c2,'d':d2,'r':r2,'t':t2}

point_j = Joints(chord,brace1)
point_k = Joints(chord,brace2)


Newton2d(point_j,point_k,[0,3/2*np.pi], prt_step = True)




# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 18:52:42 2022

@author: 63791
"""
import numpy as np
from rotation import stress_transform
from numpy import array
import pandas as pd
from itertools import product
from math import isclose
from windpower import Joints, Newton2d


def array2mat(array):
    mat = \
        [[array[0], array[3], array[5]],
         [array[3], array[1], array[4]],
         [array[5], array[4], array[2]]] 
    return np.matrix(mat)
    
def mat2array(mat):
    array = [mat[0,0], mat[1,1], mat[2,2], mat[0,1], mat[1,2], mat[0,2]]
    return np.array(array)  

def dist(series):
    return np.sqrt(np.sum(series["x"]**2 + series["y"]**2 + series["z"]**2 ))

def local_stress_cal(series):
    stresses = series.apply(array2mat)
    local_vec = list(local_axis[series.name].values())  #series.name = rad 每一列同一個local_axis
    return np.matrix([ mat2array(stress_transform(stress, local_vec)) for stress in stresses]).T



# =============================================================================
# 1.input
# =============================================================================
# 弦桿中心線
a = [-1421.00, 16246.00, -40360.00]
b = [-1421.00, 16246.00,-29660.00]
R = 1420
T = 105

# # 斜撐中心線1
# c1 = [-2064.00, 14980.00, -34128.00]
# d1 = [-3709.00, 11079.00, -30614.00]
r = 717
t = 32

# # 斜撐中心線2
# c2 = [-568.00, 15111.00, -34078.00]
# d2 = [1729.00, 11555.00, -30564.00]

# # 其他
tw = 10 #焊道腳長
N = 9 # 每一象限幾等分
chord = {'a':a,'b':b,'R':R,'T':T}                                                      #chord
data = pd.read_excel('.\data\data.xlsx',index_col=0)                                   #a,b,c,d data
coor = pd.read_excel('ANSYS_NODE_XYZ.xlsx',index_col=0)                                # coordinate ane NumberOfNode             
numberOfBrace = 4
# =============================================================================
# 2.data ETL (Extract-Transform-Load)
# =============================================================================
# **Data資料與應力資料合併**
#計算多組Joints data，並合併
for i in range(1, numberOfBrace+1):
    exec("c{0}, d{0} = data.filter(regex='({0}$)', axis=0).values".format(i))          #inpit c,d
    exec("brace"+ str(i) +" = {'c':c" + str(i) + ",'d':d"+ str(i) +",'r':r,'t':t}")    #brace
    exec("b{0} = Joints(chord,brace{0})".format(i))                                    #bx =Jointsx
    exec("b{0}.cal_point()".format(i))
    exec("df_{0} = b{0}.df".format(i))
    exec("df_{0}['braceNumber'] = {0}".format(i))                                      #add feature braceNumber
    
Data = pd.concat([eval("df_{0}".format(i)) for i in range(1, numberOfBrace+1)], ignore_index=True)  #concat  numberOfBrace+1 Brace data
table = pd.pivot_table(Data,['Local_切向', 'Local_徑向', 'Local_法向'],['braceNumber','種類','內外側','rad'],[])    #Local coordinate


# Find Data's node NodeNumber(對應應力用)
for index,values in coor.iterrows():
    x,y,z = values
    mask = \
    (Data['x'].apply(lambda s: isclose(s,x,abs_tol=0.9))) & \
    (Data['y'].apply(lambda s: isclose(s,y,abs_tol=0.9))) & \
    (Data['z'].apply(lambda s: isclose(s,z,abs_tol=0.9)))  
    Data.loc[Data[mask].index,'NodeNumber'] = index 
    
filter_na = Data['NodeNumber'].dropna().index     # 先刪除toe(沒有stress資料)
for i in range(1,31):
    exec("df_{0}=pd.read_table(r'.\Final code\TESTMASK_{0}.txt',sep='\s+',header=None,index_col=0)".format(i))   #read stresses
    exec("df_{0}.columns = ['X','Y','Z','XY','YZ','XZ']".format(i))
    exec("df_dict_{0} = df_{0}.to_dict('index')".format(i))
    exec("Data['t{0}'] = Data.loc[filter_na,'NodeNumber'].apply(lambda num : list(df_dict_{0}[num].values())) ".format(i))   #concat stresses and Data
    



output = {}    # global influence matrix
for number, cb, position in product([1,2,3,4], ['弦桿','斜撐'],['內側','外側']):
    Data_filter = Data.query("braceNumber == @number and  種類 == @cb and 內外側 == @position")
    group = Data_filter.groupby('rad')[['x','y','z']]
    La = (group.aggregate(lambda s: s.iloc[0]- s.iloc[1]).apply(dist,axis=1)) ; La = pd.DataFrame(La)  #La
    Lb = (group.aggregate(lambda s: s.iloc[0]- s.iloc[2]).apply(dist,axis=1)) ; Lb = pd.DataFrame(Lb)  #Lb
    
   #資料格式整理
    for i in range(1,31):
        La["t{0}".format(i)] = La[0]; Lb["t{0}".format(i)] = Lb[0]       
    La = La.drop([0], axis=1); Lb = Lb.drop([0], axis=1)   
    col = [ "t{0}".format(i) for i in range(1,31) ]
    
    group2 = Data_filter.groupby('rad')[col]
    sigma_a = group2.aggregate(lambda s: array(s.iloc[1]))  #sigma_a
    sigma_b = group2.aggregate(lambda s: array(s.iloc[2]))  #sigma_b
    output[number,cb,position] = sigma_b + (sigma_a - sigma_b) * ((Lb-La) + La) / (Lb-La)   #extrapolation

final = {} # local influence matrix
for key,df in output.items():
    local_axis =  (table.loc[key]).to_dict('index')
    display = df.apply(local_stress_cal, axis=1)
    for theta, value in display.items():
        final[key+(theta,)] = value

    
    









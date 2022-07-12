# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:03:10 2022

@author: 63791
"""
import unittest
import windpower as wp
import numpy as np
from rotation import stress_transform
from itertools import product
import pandas as pd
from numpy import array
from math import isclose

def array2mat(array):
    mat = \
        [[array[0], array[3], array[5]],
         [array[3], array[1], array[4]],
         [array[5], array[4], array[2]]] 
    return np.array(mat)
    
def mat2array(mat):
    array = [mat[0,0], mat[1,1], mat[2,2], mat[0,1], mat[1,2], mat[0,2]]
    return np.array(array)  

def dist(series):
    return np.sqrt(np.sum(series["x"]**2 + series["y"]**2 + series["z"]**2 ))

def local_stress_cal(series,local_axis):
    stresses = series.apply(array2mat)
    local_vec = list(local_axis[series.name].values())  #series.name = rad 每一列同一個local_axis
    return np.array([ mat2array(stress_transform(stress, local_vec) ) for stress in stresses])

class AddTestCase(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_localIM(self):
        a = [-1421.00, 16246.00, -40360.00]
        b = [-1421.00, 16246.00,-29660.00]
        R = 1420
        T = 105
        c1 = [-2064.00, 14980.00, -34128.00]
        d1 = [-3709.00, 11079.00, -30614.00]
        r = 717
        t = 32
        chord = {'a':a,'b':b,'R':R,'T':T}
        brace1 = {'c':c1,'d':d1 ,'r':r,'t':t}
        b1 = wp.Joints(chord, brace1)
        b1.cal_point()
        b1.read_ANSYS_NODE()
        b1.read_stress_data()
        b1.local_cal()
        im1 = b1.local_IM
        
        df = b1.df
        table = pd.pivot_table(df,['Local_切向', 'Local_徑向', 'Local_法向'],['種類','內外側','rad'],[])
        
        im2 = {}    
        for cb, position in product(['弦桿','斜撐'],['內側','外側']):
            df_filter = df.query("種類 == @cb and 內外側 == @position")
            group = df_filter.groupby('rad')[['x','y','z']]
            La = (group.aggregate(lambda s: s.iloc[0]- s.iloc[1]).apply(dist,axis=1)) ; La = pd.DataFrame(La)  #La
            Lb = (group.aggregate(lambda s: s.iloc[0]- s.iloc[2]).apply(dist,axis=1)) ; Lb = pd.DataFrame(Lb)  #Lb
                        
            col = [ "t{0}".format(i) for i in range(1,31) ]            
            group2 = df_filter.groupby('rad')[col]
            sigma_a = group2.aggregate(lambda s: array(s.iloc[1]))  #sigma_a
            sigma_b = group2.aggregate(lambda s: array(s.iloc[2]))  #sigma_b 
            local_axis =  (table.loc[(cb,position)]).to_dict('index')
            
            sigma_a_local = sigma_a.apply(lambda s: local_stress_cal(s,local_axis), axis=1)
            sigma_b_local = sigma_b.apply(lambda s: local_stress_cal(s,local_axis), axis=1)
            for theta in sigma_a_local.index:
                im2[(cb, position, theta)] = (sigma_b_local.loc[theta] + \
                    (sigma_a_local.loc[theta] - sigma_b_local.loc[theta]) * ( ((Lb.loc[theta]-La.loc[theta]) + La.loc[theta]) / (Lb.loc[theta]-La.loc[theta]) ).values[0]).T         
        boolean = np.array([ np.allclose(im1[key],im2[key]) for key in im1.keys()])
        self.assertTrue( boolean.all() ,"Local influence matrix is different from which Rotated first and extrapolated lastly.")
    
    def test_Newton2d(self):
        # 弦桿中心線
        a = [-1421.00, 16246.00, -40360.00]
        b = [-1421.00, 16246.00,-29660.00]
        R = 1420
        T = 105
        # # 斜撐中心線1
        c1 = [-2064.00, 14980.00, -34128.00]
        d1 = [-3709.00, 11079.00, -30614.00]
        r = 717
        t = 32
        # # 斜撐中心線2
        c2 = [-568.00, 15111.00, -34078.00]
        d2 = [1729.00, 11555.00, -30564.00]

        # # 其他
        tw = 10 #焊道腳長
        N = 9 # 每一象限幾等分
        chord = {'a':a,'b':b,'R':R,'T':T}    #chord
        brace1 = {'c':c1,'d':d1 ,'r':r,'t':t}
        brace2 = {'c':c2,'d':d2 ,'r':r,'t':t}
        b1 = wp.Joints(chord,brace1, tw, N); b1.cal_point()
        b2 = wp.Joints(chord,brace2, tw, N); b2.cal_point()
        x1, x2, distance = wp.Newton2d(b1,b2)
        
        boolean = array( list(
            map( lambda x, y: isclose(x, y, rel_tol=0.01),
            [x1, x2, distance],
            [0.0258609, 3.1670503, 106.42])))
        self.assertTrue( boolean.all() ,"兩接合線最短直線距離有誤")
        
        
if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(AddTestCase('test_localIM'))
    suite.addTest(AddTestCase('test_Newton2d'))
    unittest.TextTestRunner(verbosity=2).run(suite)
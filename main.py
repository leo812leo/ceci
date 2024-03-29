# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 18:52:42 2022

@author: 63791
"""
import windpower as wp
# =============================================================================
# 1.input
# =============================================================================
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

# =============================================================================
# 計算
# =============================================================================
b1 = wp.Joints(chord,brace1)
b1.cal_point()
b2 = wp.Joints(chord,brace2)
b2.cal_point()

wp.distance_check(b1,b2)
b1.data_to_ansys()

#b1.to_excel(1,'points1.xlsx')
#b2.to_excel(1,'points1.xlsx')

b1.read_ansys_data()
b1.im_cal()
b1.to_excel(2,'local_IM.xlsx')
b1.to_excel(3,'global_IM.xlsx')
b1.data_plot()


wp.Newton2d(b1, b2)



import os
os.system('python test.py')
"""
module docstring
"""
# pylint: disable=invalid-name
# pylint: disable=abstract-class-instantiated
# 系統
import os
from collections import defaultdict
from itertools import chain,accumulate,product
from math import isclose
# 第三方
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts.charts import Line, Page, Timeline
import pyecharts.options as opts
# own
from function import Newton, unit_vector, num,sign_convention, local_stress_cal
from fig import plot_figure


class Joints():
    """
    class docstring
    """
# =============================================================================
# 屬性(Attribute)
# =============================================================================
    def __init__(self, chord, brace, tw=10, N=9):
        self.chord = chord  # chord座標
        self.brace = brace  # brace座標
        self.N = N
        self.tw = tw
        self.df = None
        self.local_IM = None
        self.global_IM = None
        a,b, self.R, self.T = self.chord.values()
        c,d, self.r, self.t = self.brace.values()
        a, b, c, d = np.array([a,b,c,d])
        self.a, self.b, self.c, self.d = a, b, c, d

        Uc = unit_vector(b-a)                                    #chord 中心線
        U_gamma = unit_vector(d-c)                               #brace 中心線
        U_alpha = unit_vector( np.cross(Uc, U_gamma) )           #chord 中心線 /brace 中心線 公垂線
        U_beta = unit_vector( np.cross(U_gamma,U_alpha) )        # (brace 平面)
        self.Uc      = Uc
        self.U_gamma = U_gamma
        self.U_alpha = U_alpha
        self.U_beta  = U_beta
        self.beta    = np.dot(Uc,U_beta)
        self.gamma   = np.dot(Uc,U_gamma)
        self.step = 0
    def __param_cal(self, order, theta, rr):
        a, _, c, _ =  self.a, self.b, self.c, self.d
        R, _, r, _ =  self.R, self.T, rr, self.t
        Uc =      self.Uc                               #chord 中心線
        U_gamma = self.U_gamma                          #brace 中心線
        U_alpha = self.U_alpha      #chord 中心線 /brace 中心線 公垂線
        U_beta =  self.U_beta   # (brace 平面)
        beta =    self.beta
        gamma =   self.gamma
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
        C2_2 = 2*r*np.cos(theta) * np.dot(c-a ,U_alpha) + \
               2* r**2 * ( np.cos(theta)**2 - np.sin(theta)**2 )
        l_2 =  1/beta * (-C1_2 + 0.5 * C2_2 * C2**-0.5 - 0.25 * C2_1**2 * C2**-1.5)
        PI_2 = -r*np.cos(theta) * U_alpha - r*np.sin(theta) * U_beta  + l_2 * U_gamma
        if order == 0:
            return C1, C2, l, PI
        if order == 1:
            return C1_1, C2_1, l_1, PI_1
        if order == 2:
            return C1_2, C2_2 ,l_2, PI_2
    def printall(self):
        """
        Returns
        -------
        None.
        """
        print(f"My chord is {self.chord}")
        print(f"My brace is {self.brace}")
    def cal_point(self):
        def f(mu,radius): #R(√(1+S^2 )μ_1+S^2/(3!√(1+S^2 ))μ_1^3+(S^4+4S^2)/(5!(1+S^2 )^1.5 ) μ_1^5)
            t1 = np.sqrt(1+S**2)
            t2 = S**2 / (6*t1)
            t3 = (S**4 + 4*S**2 ) / (120* t1**3)
            return radius*( t1 * mu + t2 * mu**3 + t3 * mu**5 )
        def iter_f(ini_val, radius, err=10**-13, maxiter = 5, prt_step = False):
            t1 = np.sqrt(1+S**2)
            l_tem = ini_val
            xx_temp = 0
            xx = l_tem / (radius * t1)
            for i in range(maxiter):
                if i != 0:
                    l = f(xx,radius)
                    xx_temp = xx
                    xx = xx + (l_tem - l) / (radius * t1)
                if prt_step is True:
                    print(f"After {i+1} iteration, the mu is updated to {xx}, f(mu) is {l}")
                if abs(xx-xx_temp)<err:
                    break
            return xx
        data = []
        a, _, c, _ =  self.a, self.b, self.c, self.d
        R, T, r, t =  self.R, self.T, self.r, self.t
        N, tw = self.N, self.tw
        #座標單位向量
        Uc =      self.Uc                               #chord 中心線
        U_gamma = self.U_gamma                          #brace 中心線
        U_alpha = self.U_alpha      #chord 中心線 /brace 中心線 公垂線
        U_beta =  self.U_beta   # (brace 平面)
        beta =    self.beta
        gamma =   self.gamma
        #長度相關
        La = 0.2 * np.sqrt(r*t)        #弦桿,斜撐a
        Lb_Brace = 0.65 * np.sqrt(r*t) #斜撐b
        #弦桿b長度計算
        Lc = 0.5 * t  #弦桿c
        Lb_Saddle = np.pi *R / 36
        Lb_Crown = 0.4 *(r*t*R*T)**0.25
        delta_L = (Lb_Crown - Lb_Saddle) /N #只有弦桿b使用到
        #分內外側
        for in_out ,rr in[('內側',r-t),('外側',r)]:
            #求角度 (與r有關)
            def diff_L(x,r=rr): #L'(x)
                C1_1 = -r*gamma*np.cos(x)
                C2_1 = 2*r*np.sin(x) * np.dot(c-a ,U_alpha) + 2* r**2 * np.cos(x) * np.sin(x)
                C2 = R**2 - ( np.dot(c-a,U_alpha) + r*np.cos(x))**2
                l_1 = 1/beta *( -C1_1 + C2_1/(2* C2**0.5 ) )
                return r*beta* np.cos(x) + gamma * l_1
            def diff2_L(x,r=rr): #L''(x)
                C2 = R**2 - ( np.dot(c-a,U_alpha) + r*np.cos(x))**2
                C1_2 = r*gamma*np.sin(x)
                C2_1 = 2*r*np.sin(x) * np.dot(c-a ,U_alpha) + 2* r**2 * np.cos(x) * np.sin(x)
                C2_2 = 2*r*np.cos(x) * np.dot(c-a ,U_alpha) + 2* r**2 *(np.cos(x)**2 - np.sin(x)**2)
                l_2 = 1/beta *( -C1_2 + 0.5*C2_2* C2**-0.5 -0.25 * C2_1**2 * C2**-1.5 )
                return -r*beta* np.sin(x) + gamma * l_2
            # theta
            theta1 = Newton(diff_L, diff2_L, np.pi/2, prt_step = False)   #90度計算極值點
            theta2 = Newton(diff_L, diff2_L, 3*np.pi/2, prt_step = False) #270度計算極值點
            # delta_theta
            delta_1 = theta1/N              #每象限N分 0~theta1 theta1~180
            delta_2 = (np.pi - theta1)/N
            delta_3 = (theta2 - np.pi)/N    #180~theta2 theta2~360
            delta_4 = (2*np.pi -theta2 )/N
            thetas = list(accumulate(
                chain([0],[delta_1]*N , [delta_2]*N , [delta_3]*N ,[delta_4]*(N-1) )))#累加
            for time,x in enumerate(thetas): #角度函數
                _, _, l, i = self.__param_cal(0,x,rr)       #0階
                _, _, l_1, _ = self.__param_cal(1,x,rr) #1階
                L= np.dot(i-a,Uc)
                p_cf = a + L * Uc #垂足 chord
                U_IF_c = (p_cf - i) /R          #徑向 chord
                U_IT = unit_vector(np.round(-rr*np.sin(x) * U_alpha
                                                + rr*np.cos(x) * U_beta
                                                + l_1 * U_gamma,4))#切向共用
                U_IN_c = np.cross(U_IF_c, U_IT) #法向 chord
                p_bf = c + l * U_gamma          #垂足 brace
                U_IF_b = (p_bf - i) /rr         #徑向 brace
                U_IN_b = -np.cross(U_IF_b, U_IT) #法向 brace *****方向內外側決定
                dataC = [Uc,U_IN_c,U_IF_c]
                dataB = [U_gamma,U_IN_b,U_IF_b]
                # 距離計算
                temp = num(N)
                lengths = list(product(['弦桿'],[tw , tw+La, tw+ Lb_Saddle +
                                               delta_L * temp[time%(N*2)] ] )) + \
                          list(product(['斜撐'],[tw , tw+La, tw+ Lb_Brace]))
                for (cb ,length),position in zip(lengths,['toe','a','b']*2):
                    #切平面向量 弦桿: (U_IN_c,chord 中心線向量)      
                    #切平面向量 斜撐: (U_IN_b,brace 中心線向量)
                    tt =  abs(np.dot(U_IN_c,Uc)) if cb == '弦桿' else abs(np.dot(U_IN_b,U_gamma))
                    S =  tt/np.sqrt(1-tt**2) if tt <=0.999999999 else 10000000000.00
                    #切平面參數rr 弦桿=R  內斜撐=r-t 外斜撐=r
                    mu = iter_f(length,R) if cb == '弦桿' else iter_f(length,rr)
                    if in_out == '內側' and cb == '弦桿':
                        sc_n1,sc_n2 = sign_convention(dataC, cb, in_out)
                        L_Coor = np.array([-R*np.cos(mu),
                                           -sc_n1 * R * np.sin(mu), -sc_n2*S*R*np.sin(mu) ])
                        vec = np.array([U_IF_c,np.cross(-U_IF_c, Uc),Uc])
                        p = p_cf
                        n = 5 if length == Lc else 1
                        output = [U_IT, U_IF_b, U_IN_b]
                    elif in_out == '外側' and cb == '弦桿':
                        sc_n1,sc_n2 = sign_convention(dataC, cb, in_out)
                        L_Coor = np.array([-R*np.cos(mu),
                                           sc_n1 * R * np.sin(mu), sc_n2*S*R*np.sin(mu) ])
                        vec =    np.array([U_IF_c,np.cross(Uc, U_IF_c),Uc])
                        p = p_cf
                        n = 5 if length == Lc else 2
                        output = [U_IT, U_IF_b, U_IN_b]
                    else:
                        sb_n1,sb_n2 = sign_convention(dataB,cb,in_out)
                        L_Coor = np.array([-rr*np.cos(mu) ,
                                           sb_n1 * rr * np.sin(mu), sb_n2*S*rr*np.sin(mu) ])
                        vec = np.array([U_IF_b,np.cross(U_gamma,U_IF_b),U_gamma])
                        p = p_bf
                        output = [U_IT,U_IF_c,U_IN_c]
                        if in_out == '內側':
                            n = 3
                        else:
                            n = 4
                    Pd = p + np.matmul(L_Coor,vec)
                    data.append([x ,cb, in_out, position, Pd, n, output, length])
                #輸出
                df = pd.DataFrame(data)
                df = pd.concat([ df.iloc[:,:4] ,
                                 df[4].apply(pd.Series),df.iloc[:,5],
                                 df[6].apply(pd.Series),
                                 df.iloc[:,-1] ], axis = 1)
                df.columns = ['rad','種類','內外側','距離',
                              'x','y','z','color','Local_切向','Local_徑向','Local_法向','弧長']
                self.df = df
                self.step = 1
    def to_excel(self,category = 1, path = 'test.xlsx'):
        '''
        Parameters
        ----------
        category : string or int
            'points' or 1 :  Feature points
            'local_IM' or 2 :  local coordinates' influence matrix
            'global_IM' or 3 :  global coordinates' influence
        path : string
            File path.
        Returns
        -------
        None.
        '''
        if category in ('points', 1) :
            self.df.to_excel(path)
        else:
            if category in ('local_IM', 2):
                generator = self.local_IM.items()
            elif category in ('global_IM', 3):
                generator = self.global_IM.items()
            with pd.ExcelWriter(path=path) as writer:
                for key, df_temp in generator:
                    df_temp = pd.DataFrame(df_temp)
                    df_temp.index = ['X','Y','Z','XY','YZ','XZ']
                    df_temp.columns = range(1,31)
                    df_temp.to_excel(writer,
                                    sheet_name="{0}-{1}-{2:.5f}(rad)".format(key[0],key[1],key[2]) )
    def plot(self):
        if isinstance(self.df,'NoneType'):
            self.cal_point()
        df = self.df
        a, b, c, d =  self.a, self.b, self.c, self.d
        R, r  =  self.R, self.r
        ax = plot_figure([a,c],[b,d],[R,r])
        df_plot = df[ df['距離'].isin(['a','b']) ]
        ax.scatter(df_plot.x, df_plot.y, df_plot.z, s=3,
                   c= df_plot.color ,cmap= plt.get_cmap('plasma'),
                   norm = plt.Normalize(vmin=1, vmax=4))
        plt.show()
    def inter(self,theta):
        _, _, _, PI = self.__param_cal(0, theta, self.r)       #0階
        _, _, _, PI_1 = self.__param_cal(1, theta, self.r) #1階
        _, _, _, PI_2 = self.__param_cal(2, theta, self.r) #1階
        return [PI,PI_1,PI_2]
    def read_ansys_data(self):
        self.read_ANSYS_NODE()
        self.read_stress_data()
    def read_ANSYS_NODE(self):
        coor= pd.read_excel(os.getcwd()+'\\ANSYS_NODE_XYZ.xlsx',index_col=0)
        df = self.df
        # Find Data's node NodeNumber(對應應力用)
        for index,values in coor.iterrows():
            x,y,z = values
            mask = \
            (df['x'].apply(lambda s: isclose(s,x,abs_tol=0.9))) & \
            (df['y'].apply(lambda s: isclose(s,y,abs_tol=0.9))) & \
            (df['z'].apply(lambda s: isclose(s,z,abs_tol=0.9)))
            df.loc[df[mask].index,'NodeNumber'] = index
        self.df = df
        return coor
    def read_stress_data(self):
        df = self.df
        stress = ''
        filter_na = df['NodeNumber'].dropna().index
        for i in range(1,31):
            stress =  pd.read_table(r'.\Final code\TESTMASK_{0}.txt'.format(i),
                                    sep=r'\s+',header=None,index_col=0)   #read stresses
            stress.columns = ['X','Y','Z','XY','YZ','XZ']
            stress = stress.to_dict('index')
            df['t'+str(i)] = df.loc[filter_na,'NodeNumber'].apply(
                lambda num : list(stress[num].values()))
        self.df = df
    def im_cal(self):
        def normalize(series):
            vec = [np.array((series.iloc[i]) ) for i in range(len(series))]
            return np.array(vec).T
        df = self.df
        table = pd.pivot_table(df,['Local_切向', 'Local_徑向', 'Local_法向'],
                               ['種類','內外側','rad'],[]) #Local coordinate
        global_IM = {}    # global influence matrix
        for cb, position in product(['弦桿','斜撐'],['內側','外側']):
            df_filter = df.query("種類 == @cb and 內外側 == @position")
            LL = pd.pivot_table(df_filter, '弧長','rad','距離')
            La = pd.DataFrame(LL['a']-LL['toe'])
            Lb = pd.DataFrame(LL['b']-LL['toe'])
           #資料格式整理
            for i in range(1,31):
                La["t{0}".format(i)] = La[0]
                Lb["t{0}".format(i)] = Lb[0]
            La = La.drop([0], axis=1)
            Lb = Lb.drop([0], axis=1)
            col = [ "t{0}".format(i) for i in range(1,31) ]
            #外插處理
            group2 = df_filter.groupby('rad')[col]
            sigma_a = group2.aggregate(lambda s: array(s.iloc[1]))  #sigma_a
            sigma_b = group2.aggregate(lambda s: array(s.iloc[2]))  #sigma_b
            global_IM[cb,position] = \
                sigma_b + (sigma_a - sigma_b) * ((Lb-La) + La) / (Lb-La)   #extrapolation
        #globalt_influence_matrix
        global_output_influence_matrix= {}
        for key,df_iter in global_IM.items():
            df_temp = df_iter.apply(normalize, axis=1)
            for theta, value in df_temp.items():
                global_output_influence_matrix[key+(theta,)] = value
        # local influence matrix
        local_output_influence_matrix = {}
        for key,df_iter in global_IM.items():
            local_axis =  (table.loc[key]).to_dict('index')
            df_temp = df_iter.apply(lambda s : local_stress_cal(s,local_axis), axis=1)
            for theta, value in df_temp.items():
                local_output_influence_matrix[key+(theta,)] = value
        self.local_IM = local_output_influence_matrix
        self.global_IM = global_output_influence_matrix
    def data_plot(self):
        df = self.df
        table = pd.pivot_table(df,['Local_切向', 'Local_徑向', 'Local_法向'],['種類','內外側','rad'],[])
        plot_data = defaultdict(dict)
        for cb, position in product(['弦桿','斜撐'],['內側','外側']):
            df_filter = df.query("種類 == @cb and 內外側 == @position")
            LL = pd.pivot_table(df_filter, '弧長','rad','距離')
            La = pd.DataFrame(LL['a']-LL['toe'])
            Lb = pd.DataFrame(LL['b']-LL['toe'])
            # group = df_filter.groupby('rad')[['x','y','z']]
            # La = (group.aggregate(lambda s: s.iloc[0]- s.iloc[1]).apply(dist,axis=1))
            # La = pd.DataFrame(La)  #La
            # Lb = (group.aggregate(lambda s: s.iloc[0]- s.iloc[2]).apply(dist,axis=1))
            # Lb = pd.DataFrame(Lb)  #Lb      
            col = [ "t{0}".format(i) for i in range(1,31) ]
            #外插處理
            group2 = df_filter.groupby('rad')[col]
            sigma_a = group2.aggregate(lambda s: array(s.iloc[1]))  #sigma_a
            sigma_b = group2.aggregate(lambda s: array(s.iloc[2]))  #sigma_b
            local_axis =  (table.loc[(cb,position)]).to_dict('index')
            #轉local
            sigma_a_local = sigma_a.apply(lambda s: local_stress_cal(s,local_axis), axis=1)
            sigma_b_local = sigma_b.apply(lambda s: local_stress_cal(s,local_axis), axis=1)
            sigma_t_local = pd.DataFrame(sigma_b_local) + \
                pd.DataFrame((sigma_a_local - sigma_b_local)) * (Lb) / (Lb-La)
            sigma_t_local = sigma_t_local[0]
            #整理出圖格式
            for theta in sigma_a_local.index:
                for step in range(1,31):
                    for direction in range(1,7):
                        plot_data[(cb, position, 'a')].\
                            setdefault(direction,{}).setdefault(step,{}).\
                            setdefault( theta, round(sigma_a_local[theta][direction-1,step-1], 3) )
                        plot_data[(cb, position, 'b')].setdefault(direction,{})\
                            .setdefault(step,{}).\
                            setdefault( theta, round(sigma_b_local[theta][direction-1,step-1], 3) )
                        plot_data[(cb, position, 't')].setdefault(direction,{}).\
                            setdefault(step,{}).\
                            setdefault( theta, round(sigma_t_local[theta][direction-1,step-1], 3) )
            #出圖
            page = Page(layout=Page.DraggablePageLayout)
            for direction in range(1,7):
                tl = Timeline(init_opts = opts.InitOpts(height="200px"))
                for step in range(1,31):
                    L =(
                        Line()
                        .add_xaxis(
                            list((df.query("種類 == @cb and 內外側 == @position")
                                  ['rad'].unique()*180/np.pi).round(0) ) )
                        .add_yaxis("a", list(plot_data[(cb, position, 'a')]
                                             [direction][step].values()) )
                        .add_yaxis("b", list(plot_data[(cb, position, 'b')]
                                             [direction][step].values()) )
                        .add_yaxis("toe", list(plot_data[(cb, position, 't')]
                                               [direction][step].values()) )
                        .set_series_opts(
                            areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
                            label_opts=opts.LabelOpts(is_show=False),
                        )
                        .set_global_opts(
                            title_opts=opts.TitleOpts(
                                title="{2}{3} - {1} - step{0}".\
                                    format(step,['\u03C3X(切向)','\u03C3Y(徑向)','\u03C3Z(法向)',
                                                 '\u03C4XY','\u03C4YZ','\u03C4XZ'][direction-1]
                                           ,cb, position)),
                            yaxis_opts=opts.AxisOpts(name = '應力'),
                            xaxis_opts=opts.AxisOpts(
                                type_ = 'value', is_show = True, name_rotate = 30,
                                interval= 30, max_ = 360, name = '角度(dergree)',
                            ),
                        ))
                    tl.add(L, "{}step".format(step))
                page.add(tl)
            page.render("{0}{1}.html".format(cb, position))

def Newton2d (point_j,point_k, ini_val = (0,np.pi), err=10**-13, maxiter = 100, prt_step = False):
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
        if prt_step is True:
            print("""After {0} iteration,\nthe solution is updated to {1},{2}, delta_x is {3}\n""".\
                  format(i,round(x1,5),round(x2,5),np.linalg.norm(delta_x)))
        elif np.linalg.norm(delta_x)<err:
            break
    return x1, x2, np.linalg.norm(diff)
def distance_check(list_of_joint) :
    
    return 
    
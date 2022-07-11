import numpy as np
from function import Newton,unit_vector,num,sign_convention
from itertools import chain,accumulate,product
from fig import plot_figure
import pandas as pd
import matplotlib.pyplot as plt


class Joints():
# =============================================================================
# 屬性(Attribute)
# =============================================================================
    def __init__(self, chord, brace, tw=10, N=9):
        self.chord = chord  # chord座標
        self.brace = brace  # brace座標 
        self.N = N
        self.tw = tw
        self.df = None
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
        
    def param_cal(self,order,theta,rr):
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
        C2_2 = 2*r*np.cos(theta) * np.dot(c-a ,U_alpha) + 2* r**2 * ( np.cos(theta)**2 - np.sin(theta)**2 )
        l_2 =  1/beta * (-C1_2 + 0.5 * C2_2 * C2**-0.5 - 0.25 * C2_1**2 * C2**-1.5)
        PI_2 = -r*np.cos(theta) * U_alpha - r*np.sin(theta) * U_beta  + l_2 * U_gamma  
        if order == 0:
            return C1, C2, l, PI       
        elif order == 1: 
            return C1_1, C2_1, l_1, PI_1
        elif order == 2: 
            return C1_2, C2_2 ,l_2, PI_2
        else:
            raise BaseException("Error")            
  
    def cal_point(self):  
        
        def f(mu,radius): #R(√(1+S^2 ) μ_1+S^2/(3!√(1+S^2 )) μ_1^3+(S^4+4S^2)/(5!(1+S^2 )^1.5 ) μ_1^5)
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
                if prt_step == True:
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
        
        La = 0.2 * np.sqrt(r*t)
        Lb_Brace = 0.65 * np.sqrt(r*t)
        
        Lb_Saddle = np.pi *R / 36
        Lb_Crown = 0.4 *(r*t*R*T)**0.25
        delta_L = (Lb_Crown - Lb_Saddle) /N
        
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
                C2_2 = 2*r*np.cos(x) * np.dot(c-a ,U_alpha) + 2* r**2 * (np.cos(x)**2 -  np.sin(x)**2)
                l_2 = 1/beta *( -C1_2 + 0.5*C2_2* C2**-0.5 -0.25 * C2_1**2 * C2**-1.5 )
                return -r*beta* np.sin(x) + gamma * l_2
            
            theta1 = Newton(diff_L, diff2_L, np.pi/2, prt_step = False)   #90度計算極值點
            theta2 = Newton(diff_L, diff2_L, 3*np.pi/2, prt_step = False) #270度計算極值點
            
            delta_1 = theta1/N; delta_2 = (np.pi - theta1)/N              #每象限N等分  0~theta1 theta1~180
            delta_3 = (theta2 - np.pi)/N; delta_4 = (2*np.pi -theta2 )/N  #           180~theta2 theta2~360
            thetas = list(accumulate(chain([0],[delta_1]*N , [delta_2]*N , [delta_3]*N ,[delta_4]*(N-1) ))) #累加
                         
            for time,x in enumerate(thetas):  
                #角度函數
               
                C1, C2, l, i = self.param_cal(0,x,rr)       #0階
                C1_1, C2_1, l_1, _ = self.param_cal(1,x,rr) #1階
                
                L= np.dot(i-a,Uc)
                p_cf = a + L * Uc #垂足 chord
                
                U_IF_c = (p_cf - i) /R          #徑向 chord
                U_IT = unit_vector(np.round(-rr*np.sin(x) * U_alpha 
                                                + rr*np.cos(x) * U_beta 
                                                + l_1 * U_gamma,4))             #切向共用 
                U_IN_c = np.cross(U_IF_c, U_IT) #法向 chord   
                 
                p_bf = c + l * U_gamma          #垂足 brace
                U_IF_b = (p_bf - i) /rr         #徑向 brace
                U_IN_b = -np.cross(U_IF_b, U_IT) #法向 brace *****方向內外側決定
                dataC = [Uc,U_IN_c,U_IF_c] 
                dataB = [U_gamma,U_IN_b,U_IF_b]         
        
                temp = num(N)
                
                lengths = list(product(['弦桿'],[tw , tw+La, tw+ Lb_Saddle +delta_L * int(temp[time%(N*2)]) ] )) + \
                          list(product(['斜撐'],[tw , tw+La, tw+ Lb_Brace]))
             
                for (cb ,length),position in zip(lengths,['toe','a','b']*2):
                    tt =  abs(np.dot(U_IN_c,Uc)) if cb == '弦桿' else abs(np.dot(U_IN_b,U_gamma))
                    S =  tt/np.sqrt(1-tt**2) if tt <=0.999999999 else 10000000000.00
                    
                    mu = iter_f(length,R) if cb == '弦桿' else iter_f(length,rr)
                    if in_out == '內側' and cb == '弦桿':
                        sc_n1,sc_n2 = sign_convention(dataC,cb,in_out) 
                        L_Coor = np.array([-R*np.cos(mu) ,-sc_n1 * R * np.sin(mu),-sc_n2*S*R*np.sin(mu) ])
                        vec = np.array([U_IF_c,np.cross(-U_IF_c,Uc),Uc])
                        p = p_cf
                        n = 1
                        output = [U_IT,U_IF_b,U_IN_b]
                        
                    elif in_out == '外側' and cb == '弦桿':
                        sc_n1,sc_n2 = sign_convention(dataC,cb,in_out)
                        L_Coor = np.array([-R*np.cos(mu) , sc_n1 * R * np.sin(mu), sc_n2*S*R*np.sin(mu) ])
                        vec =    np.array([U_IF_c,np.cross(Uc, U_IF_c),Uc]) 
                        p = p_cf
                        n = 2
                        output = [U_IT,U_IF_b,U_IN_b]
                        
                    else:
                        sb_n1,sb_n2 = sign_convention(dataB,cb,in_out)
                        L_Coor = np.array([-rr*np.cos(mu) , sb_n1 * rr * np.sin(mu), sb_n2*S*rr*np.sin(mu) ])
                        vec =    np.array([U_IF_b,np.cross(U_gamma,U_IF_b),U_gamma]) 
                        p = p_bf
                        output = [U_IT,U_IF_c,U_IN_c]
                        
                        if in_out == '內側':
                            n = 3
                        else:
                            n = 4
                    Pd = p + np.matmul(L_Coor,vec)
                    data.append([x ,cb, in_out, position, Pd,n,output])  
                #輸出
                df = pd.DataFrame(data)
                df = pd.concat([df.iloc[:,:4] ,df[4].apply(pd.Series),df.iloc[:,5],df[6].apply(pd.Series)], axis = 1)
                df.columns = ['rad','種類','內外側','距離','x','y','z','color','Local_切向','Local_徑向','Local_法向']
                
                self.df = df 
                
    def to_excel(self,path):
        self.df.to_excel(path + '.xlsx')
        
    def plot(self):  
        if (type(self.df) == 'NoneType') or (self.df == None) :
            self.cal_point()
        df = self.df
        a, b, c, d =  self.a, self.b, self.c, self.d
        R, r  =  self.R, self.r     
        ax = plot_figure([a,c],[b,d],[R,r])
        df_plot = df[ df['距離'].isin(['a','b']) ]
        ax.scatter(df_plot.x, df_plot.y, df_plot.z, s=3, c= df_plot.color ,cmap= plt.get_cmap('plasma'),norm = plt.Normalize(vmin=1, vmax=4))
        plt.show()

    def printall(self):
        print(f"My chord is {self.chord}")    
        print(f"My brace is {self.brace}")
        
    def inter(self,theta):
        C1, C2, l, PI = self.param_cal(0,theta)       #0階
        C1_1, C2_1, l_1, PI_1 = self.param_cal(1,theta) #1階 
        C1_2, C2_2, l_2, PI_2 = self.param_cal(2,theta) #1階 
        return [PI,PI_1,PI_2]

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
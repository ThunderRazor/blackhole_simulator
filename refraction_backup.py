'''
Copyright 2023 by ThunderRazor (AKA SunBlade#0673).
All rights reserved.
'''

import numpy as np
import cv2


''' canvas '''
def canvas_add_blackhole_lorentzian2(C,X,Y,Center=[0.0,0.0],Radius=6000,Mass=1000):
    R = ((X-Center[0])**2 + (Y-Center[1])**2)**0.5
    C = 1-Mass/(1+(R/Radius)**2)
    C *= (C>0)
    #C = relative_c(Mass,R)
    return C

def relative_c_lorentzian2(Position=[0.0,0.0],Center=[0.0,0.0],Radius=6000,Mass=1000):
    dX,dY = Position[0]-Center[0],Position[1]-Center[1]
    unit_R = unit2([dX,dY])
    R = (dX**2+dY**2)**0.5
    C = 1-Mass/(1+(R/Radius)**2)
    C *= (C>0)
    grad_c = (Mass/(1+(R/Radius)**2))*(2*R/Radius**2)*(C>0)
    grad_c = mult2(grad_c,unit_R)
    return C,grad_c

def canvas_add_blackhole2(C,X,Y,Center=[0.0,0.0],Radius=6000,Mass=1000):
    R = ((X-Center[0])**2 + (Y-Center[1])**2)**0.5
    C = (1-2*Mass/(R+(R==0)))*(R>0)
    C *= (C>0)
    C = C**0.5
    if np.max(np.abs(np.real(C)))>1:
        C = C-1
    else:
    #C = relative_c(Mass,R)
        C = C**0.25
    return C

def relative_c2(Position=[0.0,0.0],Center=[0.0,0.0],Radius=6000,Mass=1000):
    dX,dY = Position[0]-Center[0],Position[1]-Center[1]
    unit_R = unit2([dX,dY])
    R = (dX**2+dY**2)**0.5
    C = (1-2*Mass/(R+(R==0)))*(R>0)
    C *= (C>0)
    C = C**0.5
    grad_c = Mass*(C>0)/(C*R**2+(C==0))
    grad_c = mult2(grad_c,unit_R)
    return C,grad_c

def canvas_init2(Shape=[500,1000],Origin=[250,500],Scale=1.0):
    C = np.ones((Shape[0],Shape[1]),type(1.0))
    x = [i for i in range(Shape[1])]
    y = [i for i in range(Shape[0])]
    X = np.array([x for i in range(Shape[0])],type(1.0))
    Y = np.array([y for i in range(Shape[1])],type(1.0))
    Y = np.transpose(Y)
    X,Y = uv2xy(X,Y,Origin,Scale)
    return C,X,Y

def canvas_draw2(C):
    S = np.shape(C)
    Image = np.zeros((S[0],S[1],3),type(1.0))
    Image[:,:,0] = 1.0-C
    Image[:,:,1] = 1.0-C
    return Image

def canvas_draw2_blackhole(C):
    S = np.shape(C)
    Image = np.zeros((S[0],S[1],3),type(1.0))
    Shadow = 1.0/3**0.5#red/pink
    Z7 = 0.7#white
    Z8 = 0.8#teal
    Z9 = 0.9#blue
    
    Image[:,:,0] = 1.0*(Shadow<=C)*(C<Z9) + (1-(C-Z9)/(1-Z9))*(Z9<=C)
    Image[:,:,1] = ((C-Shadow)/(Z7-Shadow))*(Shadow<=C)*(C<Z7) + 1.0*(Z7<=C)*(C<Z8) + (1-(C-Z8)/(Z9-Z8))*(Z8<=C)*(C<Z9)
    Image[:,:,2] = (C/Shadow)*(C<Shadow) + 1.0*(Shadow<=C)*(C<Z7) + (1-(C-Z7)/(Z8-Z7))*(Z7<=C)*(C<Z8)
    return Image

def xy2uv(x,y,Origin,Scale=1.0):
    u,v = (x/Scale)+Origin[1],(-y/Scale)+Origin[0]
    return u,v

def uv2xy(u,v,Origin,Scale=1.0):
    x,y = (u-Origin[1])*Scale,-(v-Origin[0])*Scale
    return x,y

''' simulation '''
def increment_circular2(grad_c,vec_c,pos_c,dt):
    vec_a,omega,norm_c,unit_c,unit_t,cos_t,sin_t = refract_get_accel2(grad_c,vec_c)

    #get new position
    ds,dth = norm_c*dt,omega*dt#increment arc length, increment theta
    dL,dH = ds*sinc(dth),ds*cosinc(dth)
    pos_dL,pos_dH = mult2(dL,unit_c),mult2(dH,unit_t)
    delta_pos = add2(pos_dL,pos_dH)
    new_pos_c = add2(pos_c,delta_pos)
    
    ##print("new pos",new_pos_c)

    #get new vector
    vec_dL,vec_dH = mult2(np.cos(dth),unit_c),mult2(np.sin(dth),unit_t)
    new_unit_c = unit2(add2(vec_dL,vec_dH))
    ##print("new unit",new_unit_c)

    #get new vec_c, assuming uniform grad_c
    delta_c = dot2(grad_c,delta_pos)
    new_norm_c = norm_c + delta_c
    new_vec_c = mult2(new_norm_c,new_unit_c)
    ##print("new vect",new_vec_c)

    return new_pos_c,new_vec_c,new_unit_c

def refract_get_accel2(grad_c,vec_c):
    norm_g,norm_c,unit_g,unit_c = norm2(grad_c),norm2(vec_c),unit2(grad_c),unit2(vec_c)
    
    #get acceleration vector
    cos_t = dot2(unit_g,unit_c)
    proj_a = mult2(cos_t,unit_c)
    unit_a = sub2(mult2(2,proj_a),unit_g)#unit_a = 2*proj_a - unit_g
    vec_a = mult2(norm_g*norm_c,unit_a)#vec_a = unit_a * mag_a;  mag_a = norm_g*norm_c
    
    #get angular speed (positive = CCW)
    Sign = np.sign(cross2(unit_g,unit_c))
    sin_t = (1.0-cos_t**2)**0.5*Sign
    omega = norm_g*sin_t
    unit_t = [-unit_c[1],unit_c[0]]#axis of curvature (90deg CCW from unit_c)
    
    return vec_a,omega,norm_c,unit_c,unit_t,cos_t,sin_t

''' projection '''
def proj2(u,v):
    '''project unit vector u on unit vector v'''
    return mult2(dot2(u,v),v)
def proj3(u,v):
    '''project unit vector u on unit vector v'''
    return mult3(dot3(u,v),v)
def proj4(u,v):
    '''project unit vector u on unit vector v'''
    return mult4(dot4(u,v),v)

''' vector cross '''
def cross2(x,y):
    [a,b],[c,d]=x,y
    return a*d-b*c
def cross3(x,y):
    [a,b,c],[d,e,f]=x,y
    return [b*f-c*e,c*d-a*f,a*e-b*d]

''' vector dot '''
def dot2(x,y):
    [a,b],[c,d]=x,y
    return a*c+b*d
def dot3(x,y):
    [a,b,c],[d,e,f]=x,y
    return a*d+b*e+c*f
def dot4(x,y):
    [a,b,c,d],[e,f,g,h]=x,y
    return a*e+b*f+c*g+d*h

''' scalar mult '''
def mult2(a,x):
    [b,c]=x
    return [b*a,c*a]
def mult3(a,x):
    [b,c,d]=x
    return [b*a,c*a,d*a]
def mult4(a,x):
    [b,c,d,e]=x
    return [b*a,c*a,d*a,e*a]

''' scalar div '''
def div2(x,a):
    [b,c]=x
    return [b/a,c/a]
def div3(x,a):
    [b,c,d]=x
    return [b/a,c/a,d/a]
def div4(x,a):
    [b,c,d,e]=x
    return [b/a,c/a,d/a,e/a]

''' vector add'''
def add2(x,y):
    [a,b],[c,d]=x,y
    return [a+c,b+d]
def add3(x,y):
    [a,b,c],[d,e,f]=x,y
    return [a+d,b+e,c+f]
def add4(x,y):
    [a,b,c,d],[e,f,g,h]=x,y
    return [a+e,b+f,c+g,d+h]

''' vector sub '''
def sub2(x,y):
    [a,b],[c,d]=x,y
    return [a-c,b-d]
def sub3(x,y):
    [a,b,c],[d,e,f]=x,y
    return [a-d,b-e,c-f]
def sub4(x,y):
    [a,b,c,d],[e,f,g,h]=x,y
    return [a-e,b-f,c-g,d-h]

''' vector unity '''
def norm2(x):
    [a,b]=x
    return (np.abs(a)**2+np.abs(b)**2)**0.5
def norm3(x):
    [a,b,c]=x
    return (np.abs(a)**2+np.abs(b)**2+np.abs(c)**2)**0.5
def norm4(x):
    [a,b,c,d]=x
    return (np.abs(a)**2+np.abs(b)**2+np.abs(c)**2+np.abs(d)**2)**0.5
def unit2(x):
    [a,b],n=x,norm2(x)
    n=n+(n==0)
    return [a/n,b/n]
def unit3(x):
    [a,b,c],n=x,norm3(x)
    n=n+(n==0)
    return [a/n,b/n,c/n]
def unit4(x):
    [a,b,c,d],n=x,norm4(x)
    n=n+(n==0)
    return [a/n,b/n,c/n,d/n]

''' math '''
def sinc(t):
    return (np.sin(t)+(t==0))/(t+(t==0))
def cosinc(t):
    return (1-np.cos(t))/(t+(t==0))

''' physics '''
def relative_c(M,R):
    return (c0()**2 - 2.0*G()*M/R)**0.5

''' constants '''
def G():
    return 6.67428E-11#m3/kg
def c0():
    return 299792458.0#m/s

def test1():
    width,height = 1000,500
    Origin = [int(height/2),int(width/2)]
    Scale = 1.0
    C,X,Y = canvas_init2(Shape=[height,width],Origin=Origin,Scale=Scale)
    C = canvas_add_blackhole_lorentzian2(C,X,Y,Center=[0.0,0.0],Radius=20,Mass=5.0)
    print(np.nanmin(C))
    X -= np.min(X)
    Y -= np.min(Y)
    X /= np.max(X)
    Y /= np.max(Y)
    #C=Y
    Image = canvas_draw2_blackhole(C)
    #Image[:,:,0] = C
    #Image[:,:,1] = Y
    #Image[:,:,2] = X

    x,y = 200,100
    print(xy2uv(x,y,Origin,Scale))
    x,y = 700,150
    print(uv2xy(x,y,Origin,Scale))
    
    g=[0.0,1.2]
    th=80
    norm_v = 100
    v=[norm_v*np.cos(np.pi*th/180),norm_v*np.sin(np.pi*th/180)]
    p=[0,100]
    #ary_p = []
    #ary_v = []
    #ary_p.append(p)
    #ary_v.append(v)
    r,c = int(height-np.round(p[1])),int(np.round(p[0]))
    if (0<=r)*(r<height)*(0<=c)*(c<width):
        Image[r,c,:] = (Image[r,c,:]+0.5)%1
        cv2.imshow("Preview",Image)
        cv2.waitKey(1)
    for i in range(1000):
        p,v,u = increment_circular2(g,v,p,0.01)
        #ary_p.append(p)
        #ary_v.append(v)
        r,c = int(height-np.round(p[1])),int(np.round(p[0]))
        if (0<=r)*(r<height)*(0<=c)*(c<width):
            Image[r,c,:] = (Image[r,c,:]+0.5)%1
            cv2.imshow("Preview",Image)
            cv2.waitKey(1)

def test2():
    width,height = 1000,500
    Origin = [int(height/2),int(width/2)]
    Scale = 1.0
    BlackHoleCenter=[0,0]
    BlackHoleMass=0.5
    BlackHoleRadius=10
    C,X,Y = canvas_init2(Shape=[height,width],Origin=Origin,Scale=Scale)
    C = canvas_add_blackhole_lorentzian2(C,X,Y,Center=BlackHoleCenter,Radius=BlackHoleRadius,Mass=BlackHoleMass)
    print(np.nanmin(C))
    X -= np.min(X)
    Y -= np.min(Y)
    X /= np.max(X)
    Y /= np.max(Y)
    #C=Y
    Image = canvas_draw2_blackhole(C)
    #Image[:,:,0] = C
    #Image[:,:,1] = Y
    #Image[:,:,2] = X

    x,y = 200,100
    print(xy2uv(x,y,Origin,Scale))
    x,y = 700,150
    print(uv2xy(x,y,Origin,Scale))

    th=45
    norm_v = 100
    p=[-400,0]

    rel_c,rel_g = relative_c_lorentzian2(Position=p,Center=BlackHoleCenter,Radius=BlackHoleRadius,Mass=BlackHoleMass)
    v=[norm_v*rel_c*np.cos(np.pi*th/180),norm_v*rel_c*np.sin(np.pi*th/180)]
    g=mult2(norm_v,rel_g)
    print("grad",g)
    #ary_p = []
    #ary_v = []
    #ary_p.append(p)
    #ary_v.append(v)
    
    c,r = xy2uv(p[0],p[1],Origin)
    r,c = int(np.round(r)),int(np.round(c))
    if (0<=r)*(r<height)*(0<=c)*(c<width):
        Image[r,c,:] = (Image[r,c,:]+0.5)%1
        cv2.imshow("Preview",Image)
        cv2.waitKey(1)
    for i in range(1000):
        p,v,u = increment_circular2(g,v,p,0.01)
        rel_c,rel_g = relative_c_lorentzian2(Position=p,Center=BlackHoleCenter,Radius=BlackHoleRadius,Mass=BlackHoleMass)
        v=mult2(norm_v*rel_c,u)
        g=mult2(norm_v,rel_g)
        c,r = xy2uv(p[0],p[1],Origin)
        try:
            r,c = int(np.round(r)),int(np.round(c))
            if (0<=r)*(r<height)*(0<=c)*(c<width):
                Image[r,c,:] = (Image[r,c,:]+0.5)%1
                cv2.imshow("Preview",Image)
                cv2.waitKey(1)
        except:
            pass


def test3():
    width,height = 1000,500
    Origin = [int(height/2),int(width/2)]
    Scale = 1.0
    BlackHoleCenter=[0,0]
    BlackHoleMass=25
    BlackHoleRadius=25
    C,X,Y = canvas_init2(Shape=[height,width],Origin=Origin,Scale=Scale)
    C = canvas_add_blackhole2(C,X,Y,Center=BlackHoleCenter,Radius=BlackHoleRadius,Mass=BlackHoleMass)
    print(np.nanmin(C))
    X -= np.min(X)
    Y -= np.min(Y)
    X /= np.max(X)
    Y /= np.max(Y)
    #C=Y
    Image = canvas_draw2_blackhole(C)
    #Image[:,:,0] = C
    #Image[:,:,1] = Y
    #Image[:,:,2] = X

    x,y = 200,100
    print(xy2uv(x,y,Origin,Scale))
    x,y = 700,150
    print(uv2xy(x,y,Origin,Scale))

    th=0
    norm_v = 200
    p=[-499,80]

    rel_c,rel_g = relative_c2(Position=p,Center=BlackHoleCenter,Radius=BlackHoleRadius,Mass=BlackHoleMass)
    v=[norm_v*rel_c*np.cos(np.pi*th/180),norm_v*rel_c*np.sin(np.pi*th/180)]
    g=mult2(norm_v,rel_g)
    print("grad",g)
    #ary_p = []
    #ary_v = []
    #ary_p.append(p)
    #ary_v.append(v)
    
    c,r = xy2uv(p[0],p[1],Origin)
    r,c = int(np.round(r)),int(np.round(c))
    if (0<=r)*(r<height)*(0<=c)*(c<width):
        Image[r,c,:] = (Image[r,c,:]+np.random.rand()/2)%1
        cv2.imshow("Preview",Image)
        cv2.waitKey(1)
    while cv2.waitKey(100)<0:
        pass
    for i in range(1000):
        p,v,u = increment_circular2(g,v,p,0.01)
        rel_c,rel_g = relative_c2(Position=p,Center=BlackHoleCenter,Radius=BlackHoleRadius,Mass=BlackHoleMass)
        v=mult2(norm_v*rel_c,u)
        g=mult2(norm_v,rel_g)
        c,r = xy2uv(p[0],p[1],Origin)
        try:
            r,c = int(np.round(r)),int(np.round(c))
            if (0<=r)*(r<height)*(0<=c)*(c<width):
                Image[r-1:r+1,c-1:c+1,:] = 1#(Image[r,c,:]+np.random.rand()/2)%1
                cv2.imshow("Preview",Image)
                #cv2.imwrite(f"simulation/{BlackHoleMass}_{th}_{i}.png",255*Image)
                cv2.waitKey(1)
        except:
            pass
    cv2.imwrite(f"simulation/{BlackHoleMass}_{th}_{i}.png",255*Image)

def test4():
    width,height = 1000,500
    Origin = [int(height/2),int(width/2)]
    Scale = 1
    BlackHoleCenter=[0,0]
    BlackHoleMass=35
    BlackHoleRadius=35
    C,X,Y = canvas_init2(Shape=[height,width],Origin=Origin,Scale=Scale)
    C = canvas_add_blackhole2(C,X,Y,Center=BlackHoleCenter,Radius=BlackHoleRadius,Mass=BlackHoleMass)
    print(np.nanmin(C))
    X -= np.min(X)
    Y -= np.min(Y)
    X /= np.max(X)
    Y /= np.max(Y)
    #C=Y
    Image = canvas_draw2_blackhole(C)*0.5
    #Image[:,:,0] = C
    #Image[:,:,1] = Y
    #Image[:,:,2] = X

    x,y = 200,100
    print(xy2uv(x,y,Origin,Scale))
    x,y = 700,150
    print(uv2xy(x,y,Origin,Scale))

    ImPrev = np.zeros((1000,height,width,3),type(1.0))

    th=[]
    norm_v = []
    p=[]#[[-499,98],[-499,98.5],[-499,99],[-499,99.5],[-499,100],[-499,100.5]]
    rel_c = []
    rel_g = []
    g = []
    c = []
    r = []
    u = []
    v = []
    r = []
    for k in range(100):
        th.append(0)
        norm_v.append(500)
        p.append([-599,-198.5539+k*0.95])
        rel_c.append(0.0)
        rel_g.append(0.0)
        g.append(0.0)
        c.append(0.0)
        r.append(0.0)
        u.append(0.0)
        v.append(0.0)
        r.append(0.0)
    #print(p)
     

    for k in range(len(p)):
        rel_c[k],rel_g[k] = relative_c2(Position=p[k],Center=BlackHoleCenter,Radius=BlackHoleRadius,Mass=BlackHoleMass)
        v[k]=[norm_v[k]*rel_c[k]*np.cos(np.pi*th[k]/180),norm_v[k]*rel_c[k]*np.sin(np.pi*th[k]/180)]
        g[k]=mult2(norm_v[k],rel_g[k])
        ##print("grad",g[k])
        #ary_p = []
        #ary_v = []
        #ary_p.append(p)
        #ary_v.append(v)
        
        c[k],r[k] = xy2uv(p[k][0],p[k][1],Origin)
        r[k],c[k] = int(np.round(r[k])),int(np.round(c[k]))
        if (0<=r[k])*(r[k]<height)*(0<=c[k])*(c[k]<width):
            Image[r[k],c[k],:] = (Image[r[k],c[k],:]+np.random.rand()/2)%1
    cv2.imshow("Preview",Image)
    cv2.waitKey(1)
    print('READY press any key to continue or Ctrl+C to abort')
    while cv2.waitKey(100)<0:
        pass
    for i in range(1000):
        for k in range(len(p)):
            p[k],v[k],u[k] = increment_circular2(g[k],v[k],p[k],0.015)
            rel_c[k],rel_g[k] = relative_c2(Position=p[k],Center=BlackHoleCenter,Radius=BlackHoleRadius,Mass=BlackHoleMass)
            v[k]=mult2(norm_v[k]*rel_c[k],u[k])
            g[k]=mult2(norm_v[k],rel_g[k])
            c[k],r[k] = xy2uv(p[k][0],p[k][1],Origin)
            try:
                r[k],c[k] = int(np.round(r[k])),int(np.round(c[k]))
                if (0<=r[k])*(r[k]<height)*(0<=c[k])*(c[k]<width):
                    Image[r[k]-0:r[k]+1,c[k]-0:c[k]+1,:] = 1#(Image[r,c,:]+np.random.rand()/2)%1
            except:
                pass
        #if (i%10)==0:
        #    print(i)
        cv2.imshow("Preview",Image)
        cv2.waitKey(1)
        #ImPrev[i,:,:,:] = Image
    print('READY press any key to continue or Ctrl+C to abort')
    while cv2.waitKey(100)<0:
        pass
    for i in range(1000):
        cv2.imshow("Preview",ImPrev[i,:,:,:])
        #cv2.imwrite(f"simulation/{BlackHoleMass}_{th}_{i}.png",255*Image)
        cv2.waitKey(30)
    cv2.imwrite(f"simulation/{BlackHoleMass}_{th}_{i}.png",255*Image)
    
if __name__ == "__main__":
    test4()










    

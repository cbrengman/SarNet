#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:30:42 2019

@author: cbrengman
"""

import numpy as np
import numpy.random as random
import math 

def get_fault_parameters(size=(224,224)):
    #Initiate random seed
    random.seed()
    
    ###########################################################################
    #DEFINE DEFAULT PARAMETERS
    
    #image dimensions (pixels equivalent to arcseconds)
    radar_los = random.choice(np.array([1,2]))         #Look direction for radar (1: ascending; 2: descending)
    image_x  = size[0]                      #final width of the image
    image_y  = size[1]                      #final height of the image
    
    #earthquake rupture parameters
    zt            = random.choice(np.arange(0.1,3,0.5)) #(0,20,1))         #depth to top of fault
    if zt <= 5:
        slip      = random.choice(np.arange(0.5,2,0.1)) #(0.5,5,0.5))         #slip range every 1 meters
    else:
        slip      = random.choice(np.arange(0.5,12,0.5))
    strike        = random.choice(np.arange(0,360,1))      #full range of strike every 30 deg (no repeats (eg 180 is mirror of 0))
    dip_ss        = random.choice(np.arange(50,91,1))      #dip for LL and RL faults
    dip_ds        = random.choice(np.arange(10,61,10))      #dip for Rev and Norm faults
    x_fcenter     = random.choice(np.arange(0,size[0],1)) #np.arange((center_x - (image_x1 / 2)),(center_x + (image_x1 / 2)), step) #fault center in x
    y_fcenter     = random.choice(np.arange(0,size[1],1)) #np.arange((center_y - (image_y1 / 2)),(center_y + (image_y1 / 2)), step) #fault center in y
    
    #define non-variable fault parameters
    Lp     = 1  #Constant
    Wp     = 1  #Constant
    FL     = random.choice(np.arange(5,20,1)) #(10,50,1)) #Idealized fault length (km)
    FW     = random.choice(np.arange(3,10,1)) #(5,30,1)) #Idealized fault width (km)
    
    ss_ds = random.choice(np.array([1,2]))
    
    if ss_ds == 1:
        dip = dip_ss
        rake = random.choice(np.array([0,180]))
    else:
        dip = dip_ds
        rake = random.choice(np.array([90,-90]))
        
    FL_arcsec = FL * 1000 / 900      #convert fault length from km to arcseconds
    FW_arcsec = FW * 1000 / 900      #downsample by 10 to match future DEM use
    zt_arcsec = zt * 1000 / 900   
                            
    #derive the step in x and y to calculate vertices
    if strike == 0:
        xstep = 0
        ystep = round(FL_arcsec / 2)
    elif strike > 0 and strike < 90:
        theta = 90 - strike
        xstep = round(FL_arcsec * np.cos(math.radians(theta)))
        ystep = round(FL_arcsec * np.sin(math.radians(theta)))
    elif strike == 90:
        xstep = round(FL_arcsec / 2)
        ystep = 0
    elif strike > 90 and strike < 180:
        theta = 180 - strike
        xstep = round(FL_arcsec * np.cos(math.radians(theta)))
        ystep = round(FL_arcsec * np.sin(math.radians(theta)))
    elif strike == 180:
        xstep = 0
        ystep = round(FL_arcsec / 2)
    elif strike > 180 and strike < 270:
        theta = 270 - strike
        xstep = round(FL_arcsec * np.cos(math.radians(theta)))
        ystep = round(FL_arcsec * np.sin(math.radians(theta)))
    elif strike == 270:
        xstep = round(FL_arcsec / 2)
        ystep = 0
    elif strike > 270 and strike < 360:
        theta = 360 - strike
        xstep = round(FL_arcsec * np.cos(math.radians(theta)))
        ystep = round(FL_arcsec * np.sin(math.radians(theta)))
    elif strike == 360: 
        xstep = 0
        ystep = round(FL_arcsec / 2)
        
    #fualt plane vertices based on strike and fault center
    vertices = np.array([[x_fcenter - xstep,x_fcenter + xstep], 
                        [y_fcenter - ystep,y_fcenter + ystep]])
    
    #Generate a datastruct in the radar line of sight for the full_image
    X = np.arange(0,image_x,1)  #width of full image
    Y = np.arange(0,image_y,1)  #height of full image
    
    Xn1, Yn1 = np.meshgrid(X,Y)        #gridded data the size of the image
    r, c     = Xn1.shape               #extract the number of rows and columns 
    Xn       = Xn1.reshape(((r * c),),order = 'F') #reshape from matrix to r*c by 1 length arrays
    Yn       = Yn1.reshape(((r * c),),order = 'F') #reshape from matrix to r*c by 1 length arrays
    
    if radar_los == 1: #"ascending":
        E = 0.6257     #Vector values for Radar LOS (ascending)
        N = 0.1152
        U = -0.7710
    if radar_los == 2: #"descending":
        E = -0.6495    #Vector values for Radar LOS (descending)
        N = 0.1179
        U = -0.7509
    
    S = np.tile(np.array([[E],[N],[U]]),[len(Xn)]) #LOS look vector for each pixel (3xnp)
    #S = np.tile(np.array([[0],[0],[1]]),[len(Xn)])
    datastruct = {
                  "S": S,
                  "X": Xn,
                  "Y": Yn
                  }
    
    faultstruct = {
                    "vertices": vertices,
                    "zt": zt_arcsec,
                    "W": FW_arcsec,
                    "dip": dip,
                    "L": FL_arcsec,
                    "strike": strike
                    }
    
    patchstruct = ver2patchconnect(faultstruct,Lp,Wp,1) #Develop patchstruct for single patch fault
    
    green = make_green(patchstruct,datastruct) #Develop greens function to create forward model Gm=d
    g1    = green[:,0] #strike-slip component
    g2    = green[:,1] #dip-slip component
    green = np.cos(math.radians(rake)) * g1 + np.sin(math.radians(rake)) * g2
    
    synth_data = green * slip #Forward model to get synthetic surface deformation data
    
    synth_data_reshape = synth_data.reshape((image_y,image_x),order='F') #reshape the data to be an image
    
    img         = synth_data_reshape
    #params      = np.array([strike,dip,rake,zt,slip,radar_los,x_fcenter,y_fcenter])
    
    return img

###############################################################################
###############################################################################
#Recreate the code from William barnhart written in Matlab
#Code returns a patchstruct based on fault parameters
#Patchstruct is a set of patches that slip on a fault
#In our case that should be 1
def ver2patchconnect(faultstruct,targetLp,Wp,faultnp):
    
    totL   = faultstruct["L"]
    Lp     = int(faultstruct["L"]/totL*targetLp)
    if Lp == 0:
        Lp = 1
    #totLp = Lp
    xt     = np.mean(faultstruct["vertices"][0,:])
    yt     = np.mean(faultstruct["vertices"][1,:])
    strike = faultstruct["strike"]
    L      = faultstruct["L"]
    W      = faultstruct["W"]
    dip    = faultstruct["dip"]
    zt     = faultstruct["zt"]
    
    if dip > 0:
        x0 = xt + W * np.cos(math.radians(dip)) * np.cos(math.radians(strike))
        y0 = yt - W * np.cos(math.radians(dip)) * np.sin(math.radians(strike))
    else:
        x0 = xt - W * np.cos(math.radians(dip)) * np.cos(math.radians(strike))
        y0 = yt + W * np.cos(math.radians(dip)) * np.sin(math.radians(strike))
        
    z0 = zt + W * np.sin(math.radians(dip))
    #xs = np.mean([xt,x0])
    #ys = np.mean([yt,y0])
    #zs = np.mean([zt,z0])
    
    dL = L / Lp
    dW = W / Wp
    dx = (xt - x0) / Wp
    dy = (yt - y0) / Wp
    
    
    for k in range(Wp):
        xtc = xt - dx * (k)
        x0c = xt - dx * (k+1)
        xsc = np.mean([x0c,xtc])
        ytc = yt - dy * (k)
        y0c = yt - dy * (k+1)
        ysc = np.mean([y0c,ytc])
        z0p = z0 - dW * (Wp - k - 1) * np.sin(math.radians(dip))
        ztp = z0 - dW * (Wp - k) * np.sin(math.radians(dip))
        zsp = np.mean([z0p,ztp])
    
        for l in range(Lp):
            #id = (l) * totLp + np.sum(Lp + Lp - l)
            lsina = (L / 2 - dL * (l)) * np.sin(math.radians(strike))
            lsinb = (L / 2 - dL * (l+1)) * np.sin(math.radians(strike))
            lcosa = (L / 2 - dL * (l)) * np.cos(math.radians(strike))
            lcosb = (L / 2 - dL * (l+1)) * np.cos(math.radians(strike))
            lsin  = (L / 2 - dL * (l + 0.5)) * np.sin(math.radians(strike))
            lcos  = (L / 2 - dL * (l + 0.5)) * np.cos(math.radians(strike))
            
            xfault = np.array([[xtc+lsina],[xtc+lsinb],[x0c+lsinb],[x0c+lsina],[xtc+lsina]])
            yfault = np.array([[ytc+lcosa],[ytc+lcosb],[y0c+lcosb],[y0c+lcosa],[ytc+lcosa]])
            zfault = np.array([[ztp],[ztp],[z0p],[z0p],[ztp]])
            
            x0p    = x0c+lsin
            y0p    = y0c+lcos
            xsp    = xsc+lsin
            ysp    = ysc+lcos
    
            patchstruct = {
                            "x0": x0p,
                            "y0": y0p,
                            "z0": z0p,
                            "xs": xsp,
                            "ys": ysp,
                            "zs": zsp,
                            "strike": strike,
                            "dip": dip,
                            "L": dL,
                            "W": dW,
                            "xfault": xfault,
                            "yfault": yfault,
                            "zfault": zfault
                            }
    return patchstruct

###############################################################################
###############################################################################
#Recreate the code from William barnhart written in Matlab
#Returns Greens function for forward modeling the displacement on the fault
def make_green(patchstruct,datastruct):
    x = datastruct["X"]
    y = datastruct["Y"]
    S = datastruct["S"]
    S = np.transpose(S)
    
    numpoints = len(x)
    
    green = np.zeros([numpoints,2])
    
    for k in range(2):
        id     = (k) * 1
        x0     = patchstruct["x0"]
        y0     = patchstruct["y0"]
        z0     = patchstruct["z0"]
        L      = patchstruct["L"]
        W      = patchstruct["W"]
        strike = patchstruct["strike"]
        dip    = patchstruct["dip"]
        
        ux,uy,uz = calc_okada(1,x-x0,y-y0,0.25,dip,z0,L,W,k,strike)
        
        green[:,id] = ux * S[:,0] + uy * S[:,1] + uz * S[:,2]
        #test = ux * S[0,:] + uy * S[1,:] + uz * S[2,:]
        
    return green
    
###############################################################################
###############################################################################
#Recreate the code from William barnhart written in Matlab
#Returns displacements based on okada faul model for dislocations in an elastic half-space
def calc_okada(U,x,y,nu,delta,d,length,W,fault_type,strike):
    if strike == 90:
        strike = 89.9
    elif strike == 45:
        strike = 44.9
    elif strike == 0:
        strike = 0.1
    if delta == 0:
        delta = 0.1
    elif delta == 90:
        delta = 89.9
    
    r = len(x)
    
    ux = np.zeros([r,])
    uy = np.zeros([r,])
    uz = np.zeros([r,])
    
    strike = -strike * np.pi / 180 + np.pi / 2
    coss   = np.cos(strike)
    sins   = np.sin(strike)
    #rot    = [[coss,-sins],[sins,coss]]
    rotx   = x * coss + y * sins
    roty   = -x * sins + y * coss
    
    L      = length/2
    delta  = delta * np.pi / 180
    
    Const = -U / (2 * np.pi)
        
    cosd  = np.cos(delta)
    sind  = np.sin(delta)
        
    p = roty * cosd + d * sind
    q = roty * sind - d * cosd
    a = 1 - 2 * nu
        
    parvec = [d, a, delta, fault_type]
        
    f1a,f2a,f3a = fBi(rotx + L, p, parvec, p, q)
    f1b,f2b,f3b = fBi(rotx + L, p - W, parvec, p, q)
    f1c,f2c,f3c = fBi(rotx - L, p, parvec, p, q)
    f1d,f2d,f3d = fBi(rotx - L, p - W, parvec, p, q)
        
    uxj = Const * (f1a - f1b - f1c + f1d)
    uyj = Const * (f2a - f2b - f2c + f2d)
    uz  = uz + Const * (f3a - f3b - f3c + f3d)
        
    ux = ux - uyj * sins + uxj * coss
    uy = uy + uxj * sins + uyj * coss
        
    return ux, uy, uz
    
###############################################################################
###############################################################################
#Recreate the code from William barnhart written in Matlab
#Returns displacements based on okada faul model for dislocations in an elastic half-space
def fBi(sig, eta, parvec, p, q):    
    #d          = parvec[0]
    a          = parvec[1]
    delta      = parvec[2]
    fault_type = parvec[3]
    
    epsn  = 1.0e-10
    cosd  = np.cos(delta)
    sind  = np.sin(delta)
    tand  = np.tan(delta)
    #cosd2 = np.cos(delta)**2
    sind2 = np.sin(delta)**2
    cssnd = np.cos(delta) * np.sin(delta)
    
    R    = np.sqrt((sig**2)+(eta**2)+(q**2))
    X    = np.sqrt((sig**2)+(q**2))
    ytil = (eta*cosd)+(q*sind)
    dtil = (eta*sind)-(q*cosd)
    
    Rdtil = R+dtil
    Rsig  = R+sig
    Reta  = R+eta
    RX    = R+X
    
    lnRdtil = np.log(Rdtil)
    lnReta  = np.log(Reta)
    
    badid          = [i for i, x in enumerate(R-eta) if x == 0]
    lnReta0        = -np.log(R-eta)
    lnReta0[badid] = math.inf * -1
    
    badid          = [i for i, x in enumerate(R) if x == 0] or [i for i, x in enumerate(Rsig) if x == 0]
    ORsig          = 1 / Rsig
    ORRsig         = 1 / (R*Rsig)
    ORRsig[badid]  = math.inf

    OReta  = 1 / Reta
    ORReta = 1 / (R*Reta)
    
    indfix = [i for i, x in enumerate(np.abs(Reta)) if x < epsn]
    if len(indfix) > 0:
        lnReta[indfix] = lnReta0[indfix]
        OReta = np.array(OReta)
        ORReta = np.array(ORReta)
        OReta[indfix]  = np.array(0) * indfix
        ORReta[indfix] = np.array(0) * indfix
    
    indfix = [i for i, x in enumerate(np.abs(Rsig)) if x < epsn]
    if len(indfix) > 0:
        #ORsig = np.array(ORsig)
        #ORRsig = np.array(ORRsig)
        ORsig[indfix]  = np.array(0) * indfix
        ORRsig[indfix] = np.array(0) * indfix
              
    theta = np.array(0) * q
    indfix = [i for i, x in enumerate(np.abs(q)) if x < epsn]
    indok  = [i for i, x in enumerate(np.abs(q)) if x > epsn]
    theta[indok] = np.arctan((sig[indok] * eta[indok]) / (q[indok] * R[indok]))
    if len(indfix) > 0:
        theta = np.array(theta)
        theta[indfix] = np.array(0) * indfix
    
    if np.abs(cosd) < epsn:
        I5 = -a * sig * sind / Rdtil
        I4 = -a * q / Rdtil
        I3 = a / 2 * (eta / Rdtil + (ytil * q) / Rdtil**2 - lnReta)
        I2 = -a * lnReta - I3
        I1 = -a / 2 * (sig * q) / Rdtil**2
    else:
        sigtemp = sig
        indfix  = [i for i, x in enumerate(np.abs(sig)) if x < epsn]
        sigtemp = np.array(sigtemp)
        sigtemp[indfix] = epsn
        I5 = a * 2 / cosd * np.arctan( (eta * (X + q * cosd) + X * RX * sind) / (sigtemp * RX * cosd))
        if len(indfix) > 0:
            I5 = np.array(I5)
            I5[indfix] = np.array(0) * indfix
        I4 = a / cosd * (lnRdtil - sind * lnReta)
        I3 = a * (1 / cosd * ytil / Rdtil - lnReta) + tand * I4
        I2 = -a * lnReta - I3
        I1 = -a / cosd * sig / Rdtil - tand * I5
        
    if fault_type == 0:
        f1 = (sig * q) * ORReta + theta + I1 * sind
        f2 = (ytil * q) * ORReta + (q * cosd) * OReta + I2 * sind
        f3 = (dtil * q) * ORReta + (q * sind) * OReta  + I4 * sind
    elif fault_type == 1:
        f1 = q / R - I3 * cssnd
        f2 = (ytil * q) * ORRsig + cosd * theta - I1 * cssnd
        f3 = (dtil * q) * ORRsig + sind * theta - I5 * cssnd
    else:
        f1 = q**2 * ORReta - I3 * sind2
        f2 = (-dtil * q), * ORRsig - sind * ((sig * q) * ORReta - theta) - I1 * sind2
        f3 = (ytil * q) * ORRsig  + cosd * ((sig * q) * ORReta- theta) - I5 * sind2
    
    return f1, f2, f3

if __name__ == "__main__":
    
    img = get_fault_parameters()
    #img_wrapped = ((((img - img.min())) * 4 * np.pi / 0.056) % (2 * np.pi)) / 2 / np.pi
    #img = cv2.normalize(img,None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
    #img = Image.fromarray(img,'L')
    #img_wrapped = cv2.normalize(img_wrapped,None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
    #img_wrapped = Image.fromarray(img_wrapped,'L')
    
    
    import matplotlib.pyplot as plt 
    plt.imshow(img,cmap='jet')
    print(np.max(img),np.min(img))
    #plt.imshow(img_wrapped,cmap='gray')
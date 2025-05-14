import numpy as np 
from scipy.special import boxcox1p
from scipy.special import inv_boxcox1p
from scipy.stats import boxcox 
from scipy.special import inv_boxcox
from __main__ import crop0, dataset
import h5py
# dataset = h5py.File('./../../../../../data/hongtao/variables_tt_re2.h5','r')
# dataset = h5py.File('./../../../../../data/hongtao/both-5-8-2020.h5','r')
lep_phi = np.array(dataset.get('lep_phi'))[0:crop0]

class Transform:
    def polar_to_cart(pt, eta, phi):
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        pz = pt*np.sinh(eta)
        return px, py, pz
        
    def cart_to_polar(px, py, pz): 
        pt = np.sqrt(px**2 + py**2)
        phi = np.arctan2(py, px)
        p1 = pt + (pt==0)
        eta = np.arcsinh(pz/p1)*(pt>0)
        return pt, eta, phi
    
    def cart1_transform(pt, eta, phi, exist):
        phi = (phi - lep_phi*exist) % (2*np.pi)
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        pz = pt*np.sinh(eta)
        return px, py, pz
        
    def inv_cart1_transform(px, py, pz, exist): 
        pt = np.sqrt(px**2 + py**2)
        phi = np.arctan2(py, px)
        p1 = pt + (pt==0)
        eta = np.arcsinh(pz/p1)*(pt>0)
        phi =  (phi + lep_phi*exist) % (2*np.pi)
        phi = phi - 2*np.pi*(phi > np.pi)
        return pt, eta, phi
    
    def cart2_transform(pt, eta, phi, exist):
        phi = (phi - lep_phi*exist) % (2*np.pi)
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        return px, py, eta
        
    def inv_cart2_transform(px, py, eta, exist): 
        lep_phi = np.array(dataset.get('lep_phi'))[0:crop0]
        pt = np.sqrt(px**2 + py**2)
        phi = np.arctan2(py, px)
        p1 = pt + (pt==0)
        # eta = np.arcsinh(pz/p1)*(pt>0)
        phi =  (phi + lep_phi*exist) % (2*np.pi)
        phi = phi - 2*np.pi*(phi > np.pi)
        return pt, eta, phi
    
    def cart3_transform(pt, eta, phi, lamb, exist):
        pt1 = boxcox1p(pt, lamb) 
#         phi = (phi - lep_phi*exist) % (2*np.pi)
        px = pt1*np.cos(phi)
        py = pt1*np.sin(phi)
        return px, py, eta
        
    def inv_cart3_transform(px, py, eta, lamb, exist): 
        pt1 = np.sqrt(px**2 + py**2)
        phi = np.arctan2(py, px)
        p1 = pt1 + (pt1==0)
#         phi =  (phi + lep_phi*exist) % (2*np.pi)
#         phi = phi - 2*np.pi*(phi > np.pi)
        pt = inv_boxcox1p(pt1, lamb)
        return pt, eta, phi
    
    def pxpy(pt, eta, phi):
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        return px, py, eta
        
    def inv_pxpy(px, py, eta): 
        pt = np.sqrt(px**2 + py**2)
        phi = np.arctan2(py, px)
        return pt, eta, phi
    
    
    def cart_box_transform(pt, eta, phi, exist):
        phi = (phi - lep_phi*exist) % (2*np.pi)
        sinl = 2/np.pi*np.arcsin(np.sin(phi))
        cosl = 2/np.pi*np.arcsin(np.cos(phi))
        ptbox, lamb = boxcox(pt+1)
        px = pt*cosl
        py = pt*sinl
        return ptbox,px,py,eta, lamb
    
    def inv_cart_box_transform(ptbox,px,py,eta, lamb, exist):
        pt = inv_boxcox(ptbox, lamb)-1
        cosl, sinl = px/pt, py/pt
        sin, cos = np.sin(np.pi/2*sinl),  np.sin(np.pi/2*cosl)
        phi = np.arctan2(sin, cos)
        phi = (phi + lep_phi*exist) % (2*np.pi)
        phi = phi - 2*np.pi*(phi > np.pi)
        return pt, eta, phi 
    
    
    def cart_linear_transform(pt, eta, phi, lamb, exist):
        phi = (phi - lep_phi*exist) % (2*np.pi)
        sinl = 2/np.pi*np.arcsin(np.sin(phi))
        cosl = 2/np.pi*np.arcsin(np.cos(phi))
        ptbox = boxcox1p(pt, lamb)
        px = pt*cosl
        py = pt*sinl
        return ptbox,px,py,eta
    
    def inv_cart_linear_transform(ptbox,px,py,eta, lamb, exist):
        pt = inv_boxcox1p(ptbox, lamb)
        cosl, sinl = px/pt, py/pt
        sin, cos = np.sin(np.pi/2*sinl),  np.sin(np.pi/2*cosl)
        phi = np.arctan2(sin, cos)
        phi = (phi + lep_phi*exist) % (2*np.pi)
        phi = phi - 2*np.pi*(phi > np.pi)
        return pt, eta, phi 
    

    def cart_pt_transform(pt, eta, phi, lamb):
        ptbox = pt # boxcox1p(pt, lamb)
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        return ptbox,px,py,eta
    
    def inv_cart_pt_transform(ptbox,px,py,eta, lamb):
        pt = ptbox # inv_boxcox1p(ptbox, lamb)
        phi = np.arctan2(py, px)
        return pt, eta, phi
    
    def cart_p_transform(pt, eta, phi, lamb):
        ptbox = boxcox1p(pt, lamb)
        px = pt*np.cos(phi)
        py = pt*np.sin(phi)
        p = ptbox*np.cosh(eta)
        return p,px,py,eta
    
    def inv_cart_p_transform(p,px,py,eta, lamb):
        ptbox = p/np.cosh(eta)
        pt = inv_boxcox1p(ptbox, lamb)
        phi = np.arctan2(py, px)
        return pt, eta, phi
    
    
    def phi_transform(arr, max0, mean, exist=None):
        arr = (arr-mean)
        arr = arr/max0/1.01/2+0.5
        z = stats.norm.ppf(arr)/2.5
        return z

    def invphi_transform(z, max0, mean, exist=None):
        arr = stats.norm.cdf(2.5*z)
        arr = (arr-0.5)*max0*1.01*2+mean
        return arr 

    def phi1_transform(arr, max0, mean, exist):
        w = (arr - lep_phi*exist) % (2*np.pi)
        x = w - 2*np.pi*(w>np.pi)
        z = x - (1-exist)*np.pi*2
        return z

    def invphi1_transform(z, max0, mean, exist):
        x = z+(1-exist)*np.pi*2
        w = x + 2*np.pi*(x<0)
        arr = (w + lep_phi*exist) % (2*np.pi)
        arr = arr - 2*np.pi*(arr > np.pi)
        return arr 


    def phi2_transform(arr, max0, mean, exist):
        w = (arr - lep_phi*exist) % (2*np.pi)
        y = w - (1-exist)*0.2
        z = y/(np.pi)
        return z

    def invphi2_transform(z, max0, mean, exist):
        y = z*np.pi
        x = y+(1-exist)*0.2
        # w = x + 2*np.pi*(x<0)
        arr = (x + lep_phi*exist) % (2*np.pi)
        arr = arr - 2*np.pi*(arr > np.pi)
        return arr 

    def phi3_transform(arr, max0, mean, exist):
        w = (arr - lep_phi*exist) % (2*np.pi)
        return (np.sin(w) - 1.2*(1-exist), np.cos(w) - 2.2*(1-exist))

    def invphi3_transform(z, max0, mean, exist):
        z1, z2 = z[0], z[1]
        w1 = z1 + 1.2*(1-exist)
        w2 = z2 + 2.2*(1-exist)
        w = np.arctan2(w1, w2)
        arr = (w + lep_phi*exist) % (2*np.pi)
        arr = arr - 2*np.pi*(arr > np.pi)
        return arr
    
    def phi4_transform(arr, max0, mean, exist):
        w = (arr - lep_phi*exist) % (2*np.pi)
        sin = 2/np.pi*np.arcsin(np.sin(w)) - 1.2*(1-exist)
        cos = 2/np.pi*np.arcsin(np.cos(w)) - 2.2*(1-exist)
        return (sin, cos)

    def invphi4_transform(z, max0, mean, exist):
        pi = np.pi
        sin, cos = z[0] + 1.2*(1-exist), z[1] + 2.2*(1-exist)
        sin0, cos0 = np.sin(pi/2*sin),  np.sin(pi/2*cos)
        w = np.arctan2(sin0, cos0)
        x = (w + lep_phi*exist) % (2*pi)
        x = x-2*np.pi*(x>pi)
        return x
    
    def phi6_transform(arr, max0, mean, exist):
        sin = 2/np.pi*np.arcsin(np.sin(arr)) - 1.2*(1-exist)
        cos = 2/np.pi*np.arcsin(np.cos(arr)) - 2.2*(1-exist)
        return (sin, cos)

    def invphi6_transform(z, max0, mean, exist):
        pi = np.pi
        sin, cos = z[0] + 1.2*(1-exist), z[1] + 2.2*(1-exist)
        sin0, cos0 = np.sin(pi/2*sin),  np.sin(pi/2*cos)
        w = np.arctan2(sin0, cos0)
        x = w
        x = x-2*np.pi*(x>pi)
        return x
    
    
    def phi5_transform(arr, max0, mean, exist):
        w = (arr - lep_phi*exist) % (2*np.pi)
        sin = np.sin(w) - 1.2*(1-exist)
        cos = np.cos(w) - 2.2*(1-exist)
        return (sin, cos)

    def invphi5_transform(z, max0, mean, exist):
        pi = np.pi
        sin, cos = z[0] + 1.2*(1-exist), z[1] + 2.2*(1-exist)
        w = np.arctan2(sin, cos)
        x = (w + lep_phi*exist) % (2*pi)
        x = x-2*np.pi*(x>pi)
        return x

#     def phi5_transform(arr, max0, mean, exist):
#         w = (arr - lep_phi*exist) % (2*np.pi)
#         w = w -2*np.pi*(w>np.pi)
#         sin = 2/np.pi*np.arcsin(np.sin(w)) - 1.2*(1-exist)
#         cos = 2/np.pi*np.arcsin(np.cos(w)) - 2.2*(1-exist)
#         return (sin, cos, w)

#     def invphi5_transform(z, max0, mean, exist):
#         pi = np.pi
#         sin, cos, w0 = z[0] + 1.2*(1-exist), z[1] + 2.2*(1-exist), z[2]
#         sin0, cos0 = np.sin(pi/2*sin),  np.sin(pi/2*cos)
#         w = np.arctan2(sin0, cos0)
#         w = (w + w0)/2
#         x = (w + lep_phi*exist) % (2*pi)
#         x = x-2*np.pi*(x>pi)
#         return x

#     def phi6_transform(arr, max0, mean, exist):
#         arr = arr - np.pi
#         w = (arr - lep_phi*exist) % (2*np.pi)
#         w = w -2*np.pi*(w>np.pi)
#         sin = 2/np.pi*np.arcsin(np.sin(w)) - 1.2*(1-exist)
#         cos = 2/np.pi*np.arcsin(np.cos(w)) - 2.2*(1-exist)
#         return (sin, cos, w)

#     def invphi6_transform(z, max0, mean, exist):
#         pi = np.pi
#         sin, cos, w0 = z[0] + 1.2*(1-exist), z[1] + 2.2*(1-exist), z[2]
#         sin0, cos0 = np.sin(pi/2*sin),  np.sin(pi/2*cos)
#         w = np.arctan2(sin0, cos0)
#         w = (w + w0)/2
#         x = (w + lep_phi*exist) % (2*pi)
#         x = x + np.pi 
#         x = x-2*np.pi*(x>pi)
#         return x

    def DL1r_transform(arr):
        return (arr+9.718866348266602)/(9.718866348266602 + 18.004295349121094)
    
    def inv_DL1r_transform(arr):
        return arr*(9.718866348266602 + 18.004295349121094) - 9.718866348266602
    
    def pt_transform(arr, max0, mean=None, exist=None):
        return arr/max0

    def invpt_transform(z, max0, mean=None, exist=None):
        return z*max0 

    def meanmax_transform(arr, max0, mean, exist=None):
        arr = arr-mean
        z = arr/max0
        return z

    def invmeanmax_transform(z, max0, mean, exist=None):
        return z*max0+mean
    
    def boxcox_transform(arr, lamb, mean=None, exist=None):
        box = boxcox1p(arr + 30, lamb)
        maxbox = np.max(box)
        z = box/maxbox
        return (z, maxbox)

    def invboxcox_transform(z, lamb, maxbox, exist=None):
        box = z*maxbox
        arr = inv_boxcox1p(box, lamb)
        return arr - 30 

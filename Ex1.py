# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 23:19:24 2016

@author: Kevin
"""
import numpy as np
import matplotlib.pyplot as plt
from pylab import e

def rhs(y):
    return -y

def euler(yn,rhs,dt):
    return yn+rhs(yn)*dt
    
def ODEsolve(Tmax,N,rhs,method,ic):
    times = np.linspace(0,Tmax,N)
    dt = (times[2]-times[0])*0.5
    yn = np.zeros(N)
    yn[0] = ic
    for i in range(0,N-1):
        yn[i+1] = method(yn[i],rhs,dt)
    return times,yn

def ConvergenceTest(Tmax,N,rhs,ic,method,order):
    Whole = np.linspace(0,Tmax,N)
    Half = np.linspace(0,Tmax,2*N)
    Quarter = np.linspace(0,Tmax,4*N)
    
    dt_w = (Whole[2]-Whole[0])*0.5
    dt_h = (Half[2]-Half[0])*0.5
    dt_q = (Quarter[2]-Quarter[0])*0.5
    
    yn_w = np.zeros(N)
    yn_h = np.zeros(2*N)
    yn_q = np.zeros(4*N)
    yn_w[0] = ic
    yn_h[0] = ic
    yn_q[0] = ic
    
    diff1 = np.zeros(N)
    diff2 = np.zeros(N)
    
    for i in range(0,4*N-1):
        yn_q[i+1] = method(yn_q[i],rhs,dt_q)
    for i in range(0,2*N-1):
        yn_h[i+1] = method(yn_h[i],rhs,dt_h)
    for i in range(0,N-1):
        yn_w[i+1] = method(yn_w[i],rhs,dt_w)
        diff1[i] = (yn_w[i+1]-yn_h[2*i+1])
        diff2[i] = (2**order)*(yn_h[2*i+1]-yn_q[4*i+1])
        
    return diff1,diff2

#Changing second term to 3,5,17,33 to view instablitlies. As dt increases, the integration appears to grow wildly.
times,yn = ODEsolve(16,5,rhs,euler,1)
plt.plot(times,yn)
plt.show()

    
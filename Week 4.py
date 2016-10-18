# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:02:48 2016

@author: Kevin
"""
#Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from pylab import e
from pylab import pi
#RHS FUNCTIONS
def rhs(y):
    return y
    
def shm(v,yn):
    """A RHS function for a simple harmonic oscillator."""
    return np.array([-(2*pi)**2*yn,v]) #First term for calculating change in velocity, second for change in y.

#STEP FUNCTIONS
def euler(yn,rhs,dt):
    """Euler's method for a single step. Returns Y_n + f(y) * dt"""
    return yn + rhs(*yn)*dt
    
def RK2(yn,rhs,dt):
    """Second order Runge-Kutta method."""
    k1 = rhs(*yn)
    k2 = rhs(*yn+(k1*dt))
    return yn + (dt*0.5) * (k1 + k2)
    
def RK4(yn,rhs,dt):
    """Fourth order Runge-Kutta method."""
    k1 = rhs(*yn)
    k2 = rhs(*yn+(k1*dt*0.5))
    k3 = rhs(*yn+(k2*dt*0.5))
    k4 = rhs(*yn+(k3*dt))
    return yn + (dt/6.) * (k1 + 2*k2 + 2*k3 +k4)

#USABLE FUNCTIONS
def ODEsolve(Tmax,N,rhs,method,ic):
    """A ODE solver function. Currently capable of solving: Multiple ODEs.
    
    This function requires both a Right-Hand-Side function (rhs) and a numerical integration step function (method).
    Tmax refers to the max time to integrate, N for the number of steps and ic refers to the inital conditions. Outputs a array
    of times and integrated values. Integrated values are always given in a 2-D array format - it may be necessary to use 
    Output[0,:].
    
    "ic" MUST be given in a array format."""
    #Setting arrays...
    times = np.linspace(0,Tmax,N+1) #N+1 to convert no. of intervals into steps.
    dt = (times[2]-times[0])*0.5
    yn = np.zeros([len(ic),N+1]) #Arrays set up for as many variables as given in the inital conditions.
    yn[:,0] = ic #Setting initial conditions to the first element of arrays. ORDER MUST BE THE SAME AS DEFINED IN RHS FUNCTION.
    
    #Main integration loop
    for i in range(0,N):
        yn[:,i+1] = method(yn[:,i],rhs,dt) #Simultaneous steps for all variables.
        
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
    
    diff1 = np.zeros(N+1)
    diff2 = np.zeros(N+1)
    
    for i in range(0,4*N-1):
        yn_q[i+1] = method(yn_q[i],rhs,dt_q)
    for i in range(0,2*N-1):
        yn_h[i+1] = method(yn_h[i],rhs,dt_h)
    for i in range(0,N-1):
        yn_w[i+1] = method(yn_w[i],rhs,dt_w)
        diff1[i] = (yn_w[i+1]-yn_h[2*i+1])
        diff2[i] = (2**order)*(yn_h[2*i+1]-yn_q[4*i+1])
        
    return diff1,diff2
    
def ConvergenceTest2(Tmax,N,rhs,ic,method,order):
    t_w,yn_w = ODEsolve(Tmax,N,rhs,method,ic)   #Whole,
    t_h,yn_h = ODEsolve(Tmax,2*N,rhs,method,ic) #Half,
    t_q,yn_q = ODEsolve(Tmax,4*N,rhs,method,ic) #Quarter Steps.
    
    for i in range(0,N): #Finding difference in position ONLY.(Assumes last column element is position)
        diff1[i] = (yn_w[-1,i+1]-yn_h[-1,2*i+1])
        diff2[i] = (2**order)*(yn_h[-1,2*i+1]-yn_q[-1,4*i+1])
    return diff1,diff2
    
"""
times,yn = ODEsolve(1,2000,shm,RK2,[0,1])
plt.plot(times,yn[1,:])
plt.plot(times,np.cos(2*pi*times))
plt.show()
"""


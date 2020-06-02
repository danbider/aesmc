#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:00:27 2020

@author: danbiderman
"""

# simulate data for a planar arm

# hand engineer a time series of torques 

import numpy as np
import torch
import matplotlib.pyplot as plt
from arm_models import arm_utils

def D(L1, L2, M1, M2, t2):
    '''inertia tensor - shape (2,2).
    a function of static parameters and the current angle t2'''
    return np.array([[L1**2*M1/3 + M2*(3*L1**2 + 3*L1*L2*np.cos(t2) + L2**2)/3, \
  L2*M2*(3*L1*np.cos(t2) + 2*L2)/6], [L2*M2*(3*L1*np.cos(t2) + 2*L2)/6, L2**2*M2/3]])
        
def h_vec(L1, L2, M2, dt1, dt2, t2):
    '''Coriolis and centripetal vector, shape (2,1).
    function of static params and instantaneus angular velocities and angle t2 '''
    return np.array([-L1*L2*M2*dt2*(2*dt1 + dt2)*np.sin(t2)/2,
                   L1*L2*M2*dt1**2*np.sin(t2)/2])

def c_vec(L1, L2, M1, M2, g, t1, t2):
    '''gravity vector, shape (2,1). a function of static parameters, current configuration of angles and gravity constant'''
    return np.array([L1*M1*g*np.cos(t1)/2 + M2*g*(2*L1*np.cos(t1) + L2*np.cos(t1 + t2))/2, 
              L2*M2*g*np.cos(t1 + t2)/2])

def sim_data(dt, num_timepoints, 
             param_dict, inits_dict, 
             sin_amp, sin_omega,
             sin_phase_diff, plot):
    
    # time vector
    t = np.linspace(start = 0, 
                    stop = (num_timepoints-1)*dt, 
                    num = num_timepoints)
    
    # define torques
    torque_1 = sin_amp * np.sin(sin_omega*t) #* (1.0/(t+1.01))
    torque_2 = sin_amp * np.sin(sin_omega*t + sin_phase_diff) #* (1.0/(t+1.01)) 
    torques = np.vstack([torque_1, torque_2])
    
    static_mat = np.array(([1,0,dt,0],
                           [0,1,0,dt], 
                           [0,0,1,0], 
                           [0,0,0,1]))
    
    z_mat = np.zeros((4,num_timepoints))
    z_mat[:,0] = [inits_dict["theta_1"],
                  inits_dict["theta_1"], 
                  inits_dict["omega_1"], 
                  inits_dict["omega_2"]]
    accel_mat = np.zeros((4,num_timepoints))
    
    for n in range(1,num_timepoints):
        # compute previous angular acceleration
        D_inv = np.linalg.inv(D(
                    param_dict["L1"], 
                    param_dict["L2"], 
                    param_dict["M1"], 
                    param_dict["M2"], 
                    z_mat[1,n-1]
                    )) # compute using previous theta_2
        
        accel_mat[2:,n] = np.dot(D_inv, torques[:,n])
        z_mat[:,n] = np.dot(static_mat, z_mat[:,n-1]) + dt*accel_mat[:,n]
        
    
    obs_x, obs_y = arm_utils.coords_from_params_mat(
                z_mat[0,:], 
                z_mat[1,:], 
                param_dict["L1"], 
                param_dict["L2"])
    
    obs = np.vstack([obs_x[:,0],
                     obs_y[:,0],
                     obs_x[:,1],
                     obs_y[:,1],
                     obs_x[:,2],
                     obs_y[:,2]])
    
    if plot:
        plt.subplot(121)
        plt.plot(z_mat.T);
        plt.subplot(122)
        plt.plot(obs.T);

    latent_arr = np.zeros((6, num_timepoints))
    latent_arr[[0,1],:] = torques
    latent_arr[2:, :] = z_mat
    
    latent_list = []
    obs_list = []
    
    for i in range(num_timepoints):
        latent_list.append(torch.tensor(
            latent_arr[:,i].reshape(1,latent_arr.shape[0]),
                dtype = torch.double))
        
        obs_list.append(torch.tensor(
                    obs[:,i].reshape(1,obs.shape[0]),
                        dtype = torch.double))
            
    return latent_list, obs_list

test = False
if test:
    
 
    # define params
    dt = 0.03
    num_timepoints = 100
    
    # set static parameters
    param_dict = {}
    param_dict["L1"] =  0.145  # length of upper arm in meters
    param_dict["L2"] =  0.284  # length of forearm in meters
    param_dict["M1"] = 0.2108  # mass of upper arm in kg
    param_dict["M2"] = 0.1938  # mass of forearm in kg
    param_dict["g"] = 0.0 # gravity
    
    # inits dict
    inits_dict = {}
    inits_dict["theta_1"] = np.pi/8.0
    inits_dict["theta_2"] = np.pi/8.0
    inits_dict["omega_1"] = 0.0
    inits_dict["omega_2"] = 0.0

    sin_amp = 0.005
    sin_omega = 4.0
    sin_phase_diff = 0.1
    num_timepoints = 100
    
    sim_lats, sim_obs = sim_data(dt, num_timepoints, 
                 param_dict, inits_dict, 
                 sin_amp, sin_omega,
                 sin_phase_diff, plot = True)        
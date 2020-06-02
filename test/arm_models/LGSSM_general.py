#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:13:58 2020

@author: danbiderman
This model script should be a flexible SSM that could be directly compared 
with a Kalman Filter/Smoother. 
The user specifies arrays for:
    initial: mean vector + covariance mat 
    transition: transition mat + covariance mat
    emission: emission mat + covariance mat
"""

import copy
import aesmc # note: imported from package
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Initial: # could be made specific for each variable, or learned 
    '''distribution for latents at t=0, i.i.d draws from normal(loc,scale)'''
    def __init__(self, loc, cov_mat):
        self.loc = torch.tensor(loc)
        self.cov_mat = torch.tensor(cov_mat)

    def __call__(self): 
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.MultivariateNormal(
            self.loc, self.cov_mat), aesmc.state.BatchShapeMode.NOT_EXPANDED)
        
class Transition(nn.Module): # 
    def __init__(self, A, Q):
        super(Transition, self).__init__()
        self.A = torch.tensor(A) # transition mat. dim latents X dim latents
        self.Q = torch.tensor(Q) # cov mat. same shape.
        self.dim_latents = self.A.shape[0]
        
    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        
        batch_size = previous_latents[-1].shape[0]
        num_particles = previous_latents[-1].shape[1]
        Q_fully_expanded = self.Q.expand(batch_size, num_particles,
                                   self.dim_latents, self.dim_latents)
        A_batch_expanded = self.A.expand(batch_size * num_particles,
                                   self.dim_latents, self.dim_latents)

        # compute Ax_{t-1}
        mean_fully_expanded = A_batch_expanded.matmul(
                            previous_latents[-1].view(
                                -1, self.dim_latents, 1)).view(
                                    batch_size,num_particles,-1)
        # return distribution x_t = Ax_{t-1} + w_t, w_t \sim N(0,Q)
        return aesmc.state.set_batch_shape_mode(
            torch.distributions.MultivariateNormal(
            mean_fully_expanded, Q_fully_expanded),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)

class Emission(nn.Module):
    """y_t = Gx_{t} + v_t, v_t \sim N(0,R)"""
    def __init__(self, C, R):
        super(Emission, self).__init__()
        self.C = torch.tensor(C) # emission mat. dim obs X dim latents
        self.R = torch.tensor(R) # cov mat. dim obs X dim obs
        self.dim_latents = self.C.shape[1]
        self.dim_obs = self.C.shape[0]
        assert(self.C.shape[0] == self.R.shape[0])

    def forward(self, latents=None, time=None, previous_observations=None):
        
        batch_size = latents[-1].shape[0]
        num_particles = latents[-1].shape[1]
        R_fully_expanded = self.R.expand(batch_size, num_particles,
                                   self.dim_obs, self.dim_obs)
        C_batch_expanded = self.C.expand(batch_size * num_particles,
                                   self.dim_obs, self.dim_latents)

        # compute Gx_{t}
        mean_fully_expanded = C_batch_expanded.matmul(
                            latents[-1].view(
                                -1, self.dim_latents, 1)).view(
                                    batch_size,num_particles,-1)

        # return distribution y_t = Gx_{t} + v_t, v_t \sim N(0,R)
        return aesmc.state.set_batch_shape_mode(
            torch.distributions.MultivariateNormal(
            mean_fully_expanded, R_fully_expanded),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)

class Bootstrap_Proposal(nn.Module):
    """This proposal is proportional to the transition.
    at step zero, the proposal should be set to the initial distribution
    at step t, the proposal should be set to the transition
    Args: ToDo: update
        scale_0, scale_t: scalars for __init__ method
        previous_latents: list of len num_timesteps, each entry is 
            torch.tensor([batch_size, num_particles, dim_latents])
        time: integer
    Returns:
        torch.distributions.Normal object. 
        at time=0, torch.tensor([batch_size, dim_latents]) 
        and at time!=0, torch.tensor([batch_size, num_particles, dim_latents])"""
    def __init__(self, init_loc, init_cov_mat, A, Q):
        super(Bootstrap_Proposal, self).__init__()
        
        self.init_loc = torch.tensor(init_loc)
        self.init_cov_mat = torch.tensor(init_cov_mat)
        self.A = torch.tensor(A) # transition mat. dim latents X dim latents
        self.Q = torch.tensor(Q) # cov mat. same shape.
        self.dim_latents = self.A.shape[0]
        
        
    def forward(self, previous_latents=None, time=None, observations=None):
        
        if time == 0: # initial. older version that works.
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.MultivariateNormal(
                                self.init_loc, self.init_cov_mat),
                aesmc.state.BatchShapeMode.NOT_EXPANDED)
 
        else: # transition
            batch_size = previous_latents[-1].shape[0]
            num_particles = previous_latents[-1].shape[1]
            Q_fully_expanded = self.Q.expand(batch_size, num_particles,
                                        self.dim_latents, self.dim_latents)
            A_batch_expanded = self.A.expand(batch_size * num_particles,
                                        self.dim_latents, self.dim_latents)
 
            # compute Ax_{t-1}
            mean_fully_expanded = A_batch_expanded.matmul(
                             previous_latents[-1].view(
                                 -1, self.dim_latents, 1)).view(
                                     batch_size,num_particles,-1)  
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.MultivariateNormal(
                mean_fully_expanded, Q_fully_expanded),
                aesmc.state.BatchShapeMode.FULLY_EXPANDED)

test = False
if test:
    initial = Initial(np.array([1.0,1.1,2.2]), 
                     np.eye(3))
    
    initial().sample()
    
    # tests for more complicated cov mats - should be psd.
    a = np.outer(np.array([1.0,0.1,1.1]),np.array([1.0,0.1,1.1]))
    
    def minor(arr,i,j):
        # ith row, jth column removed
        return arr[np.array(list(range(i))+list(range(i+1,arr.shape[0])))[:,np.newaxis],
                   np.array(list(range(j))+list(range(j+1,arr.shape[1])))]

    # check if PSD
    a_min = minor(a,0,2)
    a_min[0,0] > 0 
    np.linalg.det(a_min) > 0 
    
    transition = Transition(np.random.random((3,3)), 0.01*np.eye(3))
    fake_prev_latents = [torch.tensor(np.random.random((7,100, 3)))]
    fake_curr_latents = [transition(fake_prev_latents, time = 2).sample()]
    fake_curr_latents[-1].shape
    
    emission = Emission(C=np.random.random((6,3)), R= 0.01 * np.eye(6))
    obs = emission(fake_curr_latents).sample()
    obs.shape
    
    proposal = Bootstrap_Proposal(np.array([1.0,1.1,2.2]), np.eye(3))
    prop_sampled = proposal(time=0).sample()
    #proposal(time)

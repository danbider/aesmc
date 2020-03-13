#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:21:56 2020

@author: danbiderman
"""

import copy
import aesmc # note: imported from package
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Initial: # could be made specific for each variable, or learned 
    '''distribution for latents at t=0, i.i.d draws from normal(loc,scale)'''
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def __call__(self): 
            return torch.distributions.MultivariateNormal(
            torch.ones(3)*self.loc, torch.eye(3)*(self.scale**2))

class Transition(nn.Module): 
    '''x \in R^3 := [f,x,\dot{x}]^T'''
    def __init__(self, m, dt, scale_force, scale_aux):
        super(Transition, self).__init__()
        self.m = m
        self.dt = dt
        self.scale_force = scale_force # std scale. square for variance.
        self.scale_aux = scale_aux # std scale. scale aux**2 should be smaller
        # than (self.dt**2)*(self.scale_force**2)
        self.diag_mat = torch.diag(torch.tensor([(self.dt**2)*
                                                 (self.scale_force**2), 
                                                 self.scale_aux**2, 
                                                 self.scale_aux**2]))

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        batch_size = previous_latents[-1].shape[0]
        num_particles = previous_latents[-1].shape[1]
        diag_expanded = self.diag_mat.expand(batch_size,num_particles,3,3)
        
        # compute a(x_{k-1}) of shape (batch_size*num_particles,3)
        # save room
        ax = torch.zeros([batch_size, num_particles, 3])
        ax[:,:,1] = previous_latents[-1][:,:,2]
        ax[:,:,2] = previous_latents[-1][:,:,0] * (1.0/self.m)
        # deterministic first order Euler integration
        mean_fully_expanded = previous_latents[-1] + self.dt * ax

        # return distribution
        return aesmc.state.set_batch_shape_mode(
            torch.distributions.MultivariateNormal(
            mean_fully_expanded, diag_expanded),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)

class Emission(nn.Module):
    """This Emission just picks x (latents[-1][:,:,1]), and adds noise"""
    def __init__(self, scale):
        super(Emission, self).__init__()
        self.scale = scale # should be nn.Parameter in the future

    def forward(self, latents=None, time=None, previous_observations=None):
        # pick first element
        mean_tensor = latents[-1][:,:,1].view(
            latents[-1].shape[0], latents[-1].shape[1], 1)

        return aesmc.state.set_batch_shape_mode(
            torch.distributions.Normal(mean_tensor, self.scale),
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
    def __init__(self, m, dt, scale_0, mu_0, scale_force, scale_aux):
        super(Bootstrap_Proposal, self).__init__()
        self.m = m
        self.dt = dt
        self.cov_0 = torch.eye(3)*(scale_0**2)
        self.mu_0 = torch.ones(3)*mu_0
        self.scale_force = scale_force # std scale. square for variance.
        self.scale_aux = scale_aux # std scale. scale aux**2 should be smaller
        # than (self.dt**2)*(self.scale_force**2)
        self.diag_mat = torch.diag(torch.tensor([(self.dt**2)*
                                                 (self.scale_force**2), 
                                                 self.scale_aux**2, 
                                                 self.scale_aux**2]))
        
    def forward(self, previous_latents=None, time=None, observations=None):
        
        #if time == 0: 
            # # initial. Note: hand coded. fully expanded for 1000.
            # # version below works equally well. 
            # return aesmc.state.set_batch_shape_mode(
            #     torch.distributions.MultivariateNormal(
            #             loc = self.mu_0.expand(observations[-1].shape[0],1000,3), 
            #             covariance_matrix = self.cov_0.expand(
            #             observations[-1].shape[0], 1000, 3, 3)),
            #     aesmc.state.BatchShapeMode.FULLY_EXPANDED)
        
        if time == 0: # initial. older version that worked.
            return aesmc.state.set_batch_shape_mode( # batch-expanded dist.
                torch.distributions.MultivariateNormal(
                        loc = self.mu_0.expand(observations[-1].shape[0],3), 
                        covariance_matrix = self.cov_0.expand(
                        observations[-1].shape[0], 3, 3)),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
 
        else: # transition
            batch_size = previous_latents[-1].shape[0]
            num_particles = previous_latents[-1].shape[1]
            diag_expanded = self.diag_mat.expand(batch_size,num_particles,3,3)
            
            # compute a(x_{k-1}) of shape (batch_size*num_particles,3)
            # save room
            ax = torch.zeros([batch_size, num_particles, 3])
            ax[:,:,1] = previous_latents[-1][:,:,2]
            ax[:,:,2] = previous_latents[-1][:,:,0] * (1.0/self.m)
            # deterministic first order Euler integration
            mean_fully_expanded = previous_latents[-1] + self.dt * ax
            
            # return distribution
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.MultivariateNormal(
                mean_fully_expanded, diag_expanded),
                aesmc.state.BatchShapeMode.FULLY_EXPANDED)


class Proposal(nn.Module):
    """This Proposal uses a linear FF mapping between (1) observations[0] -> mu[0]
    and {previous_latents[t-1], observations[t]} -> mu[t].
    The weights and biases of each mapping could be learned. 
    Args:
        scale_0, scale_t: scalars for __init__ method
        previous_latents: list of len num_timesteps, each entry is 
            torch.tensor([batch_size, num_particles, dim_latents])
        time: integer
        observations: list of len num_timesteps. each entry is a
        torch.tensor([batch_size, dim_observations]
    Returns:
        torch.distributions.Normal object. 
        at time=0, torch.tensor([batch_size, dim_latents]) 
        and at time!=0, torch.tensor([batch_size, num_particles, dim_latents])"""
    
    def __init__(self, scale_0, scale_t):
        super(Proposal, self).__init__()
        self.scale_0 = scale_0
        self.scale_t = scale_t
        self.lin_0 = nn.Sequential(
                        nn.Linear(6, 2),
                        nn.ReLU()) # observations[0] -> mu[0]
        self.lin_t = nn.Sequential(
                        nn.Linear(14, 2), # SHOULD BE 6,2 IN EULER
                        nn.ReLU()) # {previous_latents[t-1], observations[t-1], observations[t]} -> mu[t]

    def forward(self, previous_latents=None, time=None, observations=None):
        if time == 0:
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.lin_0(observations[0]),
                    scale=self.scale_0),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        else:
            if time == 1:
                self.batch_size, self.num_particles, self.dim_latents = previous_latents[-1].shape
            expanded_obs_prev = aesmc.state.expand_observation(observations[time-1], self.num_particles) # expand current obs
            expanded_obs = aesmc.state.expand_observation(observations[time], self.num_particles) # expand current obs
            # concat previous latents with current expanded observation. then squeeze to batch_expanded mode
            # i.e., [batch_size*num_particles, dim_latent+dim_observation]
            # to apply a linear layer.
            concat_squeezed = torch.cat([previous_latents[-1],
                                         expanded_obs_prev,
                        expanded_obs
                        ], 
                        dim=2).view(-1, previous_latents[-1].shape[2]+
                                    expanded_obs.shape[2] +
                                    expanded_obs_prev.shape[2] )
            activ = self.lin_t(concat_squeezed)
            mu_t = activ.view(self.batch_size, self.num_particles, self.dim_latents)
            
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=mu_t,
                    scale=self.scale_t),
                aesmc.state.BatchShapeMode.FULLY_EXPANDED)        

class TrainingStats(object):
    def __init__(self, L1_true, L2_true,
                 num_timesteps,
                 logging_interval=100):
        self.L1_true = L1_true
        self.L2_true = L2_true
        self.logging_interval = logging_interval
        self.curr_L1 = []
        self.curr_L2 = []
        self.loss = []
  
    def __call__(self, epoch_idx, epoch_iteration_idx, loss, initial,
                 transition, emission, proposal):
        if epoch_iteration_idx % self.logging_interval == 0:
            print('Iteration {}: Loss = {}'.format(epoch_iteration_idx, loss))
            self.curr_L1.append(emission.L1.item())
            self.curr_L2.append(emission.L2.item()) # seemed to be better than detach().numpy()
            self.loss.append(loss)
            # print(np.round(np.linalg.norm(
            #     np.array([emission.L1.detach().numpy(),emission.L2.detach().numpy()])-
            #     np.array([self.L1_true, self.L2_true])),2))
            
def sim_data(dt, num_timepoints, m, 
             sin_amp, sin_omega, 
             cos_amp, cos_omega, 
             obs_noise_scale, plot):
    t = np.linspace(start = 0, 
                    stop = (num_timepoints-1)*dt, 
                    num = num_timepoints)
    
    force = sin_amp * np.sin(sin_omega*t) + \
            cos_amp * np.cos(cos_omega*t +0.01)
    
    force = force*t
            
    #force = np.concatenate([-0.1*np.ones(20),0.15*np.ones(40), -0.2*np.ones(40)])
    
    x_vec = np.zeros((2,num_timepoints)) # [x(t), xdot(t)]^T
    
    # Euler
    for i in range(1,num_timepoints):
        ax = np.array([x_vec[1,i-1],force[i-1]/m])
        x_vec[:,i] = x_vec[:,i-1] + dt*ax
    
    # noisy obs is position
    obs = x_vec[0,:] +  np.random.normal(0, obs_noise_scale, num_timepoints)
    
    latent_arr = np.zeros((3, num_timepoints))
    latent_arr[0,:] = force
    latent_arr[1:, :] = x_vec
    
    latent_list = []
    obs_list = []
    for i in range(num_timepoints):
        latent_list.append(torch.tensor(latent_arr[:,i].reshape(1,3),
                                        dtype = torch.float))
        obs_list.append(torch.tensor(obs[i].reshape(1,1),
                                     dtype = torch.float))
        
    if plot:
        plt.plot(force, label = 'f')
        plt.plot(x_vec[0,:], label = 'coord')
        plt.plot(x_vec[1,:], label = 'velocity')
        plt.scatter(np.arange(num_timepoints), obs, label = 'obs', marker = 'x')
        plt.legend();
        
    return latent_list, obs_list           
            
test = False
if test:
    # ToDo: keep exploring the dimensions of things and how my reshape effects
    # the forward kin function, etc.
    
    
    batch_size = 10
    num_particles = 100
    dt = 0.03
    scale=1
    
    initial = Initial(0.7, 0.5)
    initial().sample()
    diag_mat = torch.diag(torch.tensor([0.03, 0.01, 0.01]))
    diag_expanded = diag_mat.expand(batch_size,num_particles,3,3)
    fake_prev_latents = [torch.tensor([3.0, 7.0, 1.2]).expand(batch_size, num_particles, 3)]
    fake_prev_latents[-1].shape
    multi_trans_dist = torch.distributions.MultivariateNormal(
        fake_prev_latents[-1], diag_expanded)
    print(multi_trans_dist)
    val = multi_trans_dist.sample()
    val.shape
    print(val[2,4,:])
    lp = multi_trans_dist.log_prob(val)
    print(lp[:2,:3])
    
    ax = torch.zeros([batch_size, num_particles, 3])
    ax[:,:,1] = fake_prev_latents[-1][:,:,2]
    ax[:,:,2] = fake_prev_latents[-1][:,:,0] * (1.0/2)
    print(ax[3, 0, :]) # should be 0, 1.2, 1.5
    # deterministic first order Euler integration
    mean_fully_expanded = fake_prev_latents[-1] + dt * ax
    #fake_prev_latents = [torch.zeros(batch_size, num_particles,2)] # list element 1
    fake_prev_latents[-1].view(-1,2).shape
    A = torch.tensor([[1, dt],[0,1]], requires_grad = False)
    
    Q = torch.tensor([[0.25*(dt**4), 0.5*(dt**3)], 
                               [ 0.5*(dt**3), dt**2]], 
                              requires_grad = False)*scale # this is 2X2
    Q.shape
    mean_batch_expanded = A.expand(batch_size*num_particles, 2,2).matmul(
                    fake_prev_latents[-1].view(-1,2,1)).view(batch_size*num_particles,-1)
    mean_fully_expanded = A.expand(batch_size*num_particles, 2,2).matmul(
                    fake_prev_latents[-1].view(-1,2,1)).view(batch_size,num_particles,-1)
   
    scale_batch_expanded = Q.expand(batch_size*num_particles, 2,2)
    torch.ones(batch_size*num_particles,2,2)
    scale_batch_expanded.shape
    little_scale = torch.tensor([1,1])
    dist = torch.distributions.MultivariateNormal(
                torch.ones(2), scale_tril = torch.diag(little_scale))
    
    G = torch.tensor([[0.5*(dt**2)], [dt]], 
                              requires_grad = False) * scale # this is for 2d cast
    dist = torch.distributions.lowrank_multivariate_normal.\
    LowRankMultivariateNormal(mean_fully_expanded, #torch.tensor([4,7], dtype = torch.float), 
                              G, torch.zeros(2)) # seems to work
    
    dist2 =  torch.distributions.lowrank_multivariate_normal.\
    LowRankMultivariateNormal(torch.tensor([4,7], dtype = torch.float), 
                              torch.tensor([1,0], dtype = torch.float).view(2,1), 
        torch.tensor([0.01, 0.01]))
    val = dist2.sample()
    print(val)
    lp = dist2.log_prob(val)
    print(lp) # because it's a multinormal, we have one scalar for log prob.
    # here's also a degenerate distribution that might work in our case
    # note that the numbers are floats, that is important.
    # first entry in cov could be either 0.0 (we'll get a nan logprob) or 0.01
    # however we get multiple, i.e., two log probs in this case
    dist3 = torch.distributions.Normal(torch.tensor([1.3, 2.7]), 
                                       torch.tensor([0.01, 1.1]))
    val = dist3.sample()
    print(val)
    lp = dist3.log_prob(val)
    print(lp)
    
    # check transiotion
    transition = Transition(dt, scale)
    dist = transition.forward(fake_prev_latents, time = 1)
    val = dist.sample()
    print(val[0,:,:])
    
    # check something inside emission
    mean = fake_prev_latents[-1][:,:,0].view(
            fake_prev_latents[-1].shape[0], fake_prev_latents[-1].shape[1], 1)
    
    # check emission
    emission = Emission(0.01)
    emission_dist = emission([val])
    emit_val = emission_dist.sample()
    print(emit_val[1,34,0])
    print(val[1,34,0]) # these two should be close
    
    m = torch.distributions.lowrank_multivariate_normal.\
        LowRankMultivariateNormal(torch.tensor([1,2]), 
                                  G, 
                                  torch.zeros(2))
    inits_dict = {}
    inits_dict["L1"] = 1.0
    inits_dict["L2"] = 1.0
    inits_dict["M1"] = 0.5
    inits_dict["M2"] = 0.5
    # note, if doesn't work, could add 1 in last dim
    inits_dict["velocity_vec"] = torch.zeros([batch_size*num_particles, 2]) # could have 1 in last dim
    inits_dict["angle_vec"] = torch.rand([batch_size*num_particles, 2])
    
    # ToDo: onsider having consts dict
    g= 0.2
    dt=0.03
    
    emission = Emission(inits_dict, dt, g, 0.01, False)
    print(list(emission.parameters()))
    big_bs = 30
    tens_ones = torch.ones(big_bs)
    D_tens_test = emission.D(tens_ones) # keep for testing Newton
    #print(D_tens_test)
    print(D_tens_test.shape) 
    
    # torch.t(torch.ones(3,1)).mm(torch.ones(3,1)*2)
    # a = torch.tensor([[1.0],[2.0]]).expand()
    # inv = torch.inverse(D_tens_test)
    # res = inv.mm(a) 
    
    # check h_vec
    
    # one timestep
    res = emission.h_vec(torch.tensor([0.6]), torch.tensor([0.2]), torch.tensor([0.3]))
    res.shape
    
    # multiple timesteps
    a = torch.ones(30)
    b = torch.ones_like(a)*2
    #b = torch.ones(31) # should fail assertion
    c = torch.ones_like(a)*3
    h_vec = emission.h_vec(a, b, c) # keep for testing Newton
    h_vec.shape
    
    # check c_vec
    a = torch.ones(30)
    b = torch.ones_like(a)*2
    #b = torch.ones(31) # should fail assertion
    c_vec = emission.c_vec(a,b) # keep for testing Newton
    c_vec.shape
    
    latent_torque = torch.ones([30,2,1])
    latent_torque.shape
    
    inst_accel = emission.Newton_2nd(latent_torque, D_tens_test, h_vec, c_vec)
    inst_accel.shape
    inst_accel.squeeze().shape
    (inst_accel.squeeze() == inst_accel[:,:,0]).detach().numpy().all()
    
    velocity = torch.ones([30,2,1])
    next_vel = emission.Euler_step(velocity, inst_accel)
    print(((next_vel - dt*inst_accel)==1).detach().numpy().all()) # should be ones
    
    #test inits
    emission.init_velocity.shape
    emission.init_velocity.view(1000,2).shape

    emission.init_velocity.view(batch_size, num_particles,2).shape
    
    # test forward sweep at t=0
    fake_prev_latents = [torch.zeros(batch_size, num_particles,2)] # element 1
    fake_prev_latents[-1][3,21,0] = 1 # change just a single value from zero
    fake_prev_latents[-1][3,15:30,0]
    # see if the difference is obseved just there
    fake_prev_latents[-1].view(batch_size*num_particles,2,1)
    len(fake_prev_latents)
    fake_prev_latents[-1].shape
    # test forward sweep at t=0
    emission.forward(fake_prev_latents, 0) 
    np.diff(emission.forward(fake_prev_latents, 0).detach().numpy())
    diffs = np.diff(emission.angles[-1].detach().numpy(),axis=0)
    (diffs==0).all() # look at row 321. it is different from prev and next row
    # seems to make sense. it goes- first batch and 100 particles, ... third batch and 100 particles
    regular_reshape = fake_prev_latents[-1].view(batch_size*num_particles,2).detach().numpy()
    # just 321 should be different
    emission.velocity[-1]
    emission.acceleration[-1] # ToDo: test out case without h_vec/c_vec.
    # test time = 1, first take another fake sample from latent
    fake_prev_latents.append(torch.zeros(batch_size, num_particles,2)) # element 2
    len(fake_prev_latents)
    emission.forward(fake_prev_latents, 1) 
    emission.angles[-1].shape
    len(emission.acceleration)
    emission.acceleration[-1]
    emission.velocity[-1]
    emission.angles[-1]
    emission.acceleration[-1].shape
    fake_prev_latents.append(torch.zeros(batch_size, num_particles,2)) # element 2
    emission.forward(fake_prev_latents, 2) 
    len(emission.acceleration)
    len(emission.velocity)
    emission.forward()
    len(emission.angles)
    emission.velocity[-1][:,:,1].shape
    emission.FW_kin_2D(angles)
    
    fake_prev_latents[-1].shape
    fake_prev_latents.squeeze(1)
    
    fake_obs = [np.zeros()]
    
    transition = Transition(1, 0.2)
    samp = aesmc.state.sample(transition(
                previous_latents=fake_prev_latents, time=2,
                previous_observations=None), batch_size, 1)
    samp.shape
    latents = aesmc.state.sample(transition, batch_size, num_particles)
    
    eye_tens = torch.eye(2).reshape(1,2,2).repeat(self.batch_size,1,1)
        
            
            
            
            
            
            
        
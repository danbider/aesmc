#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:17:45 2020

@author: danbiderman
A model script for inference of the latent joint angles in a 2D-FW-KIN model. 
For the definition of the objects, I followed the lgssm and gaussian models.
"""

import copy
import aesmc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def FW_kin_2D(L1, L2, latents):
    '''Args:
            L1,L2: scalar quantities or nn.Parameter
            latents: torch.tensor [batch_size, num_particles, n_latent_dim]
       Returns:
            coords: torch.tensor [batch_size, num_particles, n_obs_dim]'''
    coords = torch.stack(
        (
        torch.zeros_like(latents[:,:,0]), # x_0
        torch.zeros_like(latents[:,:,0]), # y_0
        L1*torch.cos(latents[:,:,0]), # x_1
        L1*torch.sin(latents[:,:,0]), # y_1
        L2*torch.cos(latents[:,:,0]+latents[:,:,1]) + 
        L1*torch.cos(latents[:,:,0]), # x_2
        L2*torch.sin(latents[:,:,0]+latents[:,:,1]) + 
        L1*torch.sin(latents[:,:,0]), # y_2
        ), 
        dim=2)
    return coords

# check (maybe save for a unit test script)
test = 0
if test:
    latents = torch.zeros(5, 3, 2)
    #latents = torch.ones(5, 3, 2) * np.pi/2
    coord_out = FW_kin_2D(1, 1, latents)
    print(coord_out)

class Initial: # distribution for t=0
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def __call__(self): # ToDo: check maybe keep Normal instead of Multivariate
        return torch.distributions.Normal(self.loc, self.scale)


class Transition(nn.Module): # 
    def __init__(self, init_mult, scale):
        super(Transition, self).__init__()
        # at a first pass - no learning, just inference (fixed AR trans.)
        # self.mult = nn.Parameter(torch.Tensor([init_mult]).squeeze())
        self.mult = init_mult # set to 1 if initialize in this manner.
        self.scale = scale

    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        return aesmc.state.set_batch_shape_mode(
            torch.distributions.Normal(
                self.mult * previous_latents[-1], self.scale),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)


class Emission(nn.Module):
    """This Emission ..."""
    def __init__(self, emission_func, L1_init, L2_init, scale, learn_static):
        super(Emission, self).__init__()
        #self.mult = nn.Parameter(torch.Tensor([init_mult]).squeeze())
        # no need for self.mult since it's not a linear emission
        # ToDo: not sure about syntax below, maybe could suffice with tensor?
        # ToDo 2: for parameter learning, set required_grad = True
        self.L1 = nn.Parameter(torch.Tensor([L1_init]).squeeze(), requires_grad = learn_static)
        self.L2 = nn.Parameter(torch.Tensor([L2_init]).squeeze(), requires_grad = learn_static)
        self.scale = scale # should be nn.Parameter in the future
        self.emission_func = emission_func

    def forward(self, latents=None, time=None, previous_observations=None):
        # FW kin function appropriately.
        mean_tensor = self.emission_func(self.L1, self.L2, latents[-1])
        return aesmc.state.set_batch_shape_mode(
            torch.distributions.Normal(mean_tensor, self.scale),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)


# the class below is from the lgssm.py script. it learns a linear mapping from latents[-1]
# and observations[current time] to propose a sample
# potential extensions: use all observations not just observations[time]
#                       relu layer instead of linear
    
#if test:
    #test the operations below
    #lin_0 = nn.Linear(6, 2)
    #out = lin_0(observations[0])
    #print(out.shape)
    #previous_latents = []
    #previous_latents.append(torch.rand([10, 100, 2]))
    #previous_latents[-1].unsqueeze(-1).shape
    
# Bootstrap proposal class
    
class Bootstrap_Proposal(nn.Module):
    """This proposal is proportional to the transition.
    at step zero, the proposal should be set to the initial distribution
    at step t, the proposal should be set to the transition
    Args:
        scale_0, scale_t: scalars for __init__ method
        previous_latents: list of len num_timesteps, each entry is 
            torch.tensor([batch_size, num_particles, dim_latents])
        time: integer
    Returns:
        torch.distributions.Normal object. 
        at time=0, torch.tensor([batch_size, dim_latents]) 
        and at time!=0, torch.tensor([batch_size, num_particles, dim_latents])"""
    def __init__(self, scale_0, mu_0, scale_t, mult_t):
        super(Bootstrap_Proposal, self).__init__()
        self.scale_0 = scale_0
        self.mu_0 = mu_0
        self.scale_t = scale_t
        self.mult_t = mult_t

    def forward(self, previous_latents=None, time=None, observations=None):
        if time == 0:
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.mu_0.expand(observations[-1].shape[0],2), # for a 2d lat
                    scale=self.scale_0),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        else:    
            return aesmc.state.set_batch_shape_mode(
            torch.distributions.Normal(
                self.mult_t * previous_latents[-1], self.scale_t),
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
        self.lin_0 = nn.Linear(6, 2) # observations[0] -> mu[0]
        self.lin_t = nn.Linear(8, 2) # {previous_latents[t-1], observations[t]} -> mu[t]

    def forward(self, previous_latents=None, time=None, observations=None):
        if time == 0:
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.lin_0(observations[0]),
                    scale=self.scale_0),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        else:
            batch_size, num_particles, dim_latents = previous_latents[-1].shape
            expanded_obs = aesmc.state.expand_observation(observations[time], num_particles) # expand current obs
            # concat previous latents with current expanded observation. then squeeze to batch_expanded mode
            # i.e., [batch_size*num_particles, dim_latent+dim_observation]
            # to apply a linear layer.
            concat_squeezed = torch.cat([previous_latents[-1], 
                        expanded_obs
                        ], 
                        dim=2).view(-1, previous_latents[-1].shape[2]+ expanded_obs.shape[2])
            activ = self.lin_t(concat_squeezed)
            mu_t = activ.view(batch_size, num_particles, dim_latents)
            
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
            print(np.round(np.linalg.norm(
                np.array([emission.L1.detach().numpy(),emission.L2.detach().numpy()])-
                np.array([self.L1_true, self.L2_true])),2))


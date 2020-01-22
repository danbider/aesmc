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
latents = torch.zeros(5, 3, 2)
#latents = torch.ones(5, 3, 2) * np.pi/2
coord_out = FW_kin_2D(1, 1, latents)
print(coord_out)

class Initial: # distribution for t=0
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def __call__(self):
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
    def __init__(self, emission_func, L1_init, L2_init, scale):
        super(Emission, self).__init__()
        #self.mult = nn.Parameter(torch.Tensor([init_mult]).squeeze())
        self.L1 = L1_init # should be nn.Parameter in the future
        self.L2 = L2_init # '' 
        self.scale = scale # ''
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
    
class Proposal(nn.Module):
    def __init__(self, scale_0, scale_t):
        super(Proposal, self).__init__()
        self.scale_0 = scale_0
        self.scale_t = scale_t
        self.lin_0 = nn.Linear(6, 2) # observations[0] -> proposed samples[0]
        self.lin_t = nn.Linear(8, 2) 
        # {previous_latents[-1], observations[time]} -> proposed samples[time]

    def forward(self, previous_latents=None, time=None, observations=None):
        if time == 0:
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.lin_0(observations[0].unsqueeze(-1)).squeeze(-1),
                    scale=self.scale_0),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        else:
            num_particles = previous_latents[-1].shape[1]
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.lin_t(torch.cat(
                        [previous_latents[-1].unsqueeze(-1),
                          observations[time].view(-1, 1, 1).expand(
                            -1, num_particles, 1
                          )],
                        dim=2
                    ).view(-1, 2)).squeeze(-1).view(-1, num_particles),
                    scale=self.scale_0),
                aesmc.state.BatchShapeMode.FULLY_EXPANDED)

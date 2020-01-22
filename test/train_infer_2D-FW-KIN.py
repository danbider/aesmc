#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:03:03 2020

@author: danbiderman
"""

import aesmc.train as train
import aesmc.losses as losses
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from arm_models import forward_kin_2D
from aesmc import statistics

initial_loc = torch.zeros(2)
initial_scale = 1 * torch.eye(2)
true_transition_mult = 1
init_transition_mult = 0
transition_scale = 0.1
true_emission_mult = 1
init_emission_mult = 0
emission_scale = 0.01
L1_true = 1.0
L2_true = 1.0
L1_init = 1.0
L2_init = 1.0
num_timesteps = 100
num_test_obs = 10
test_inference_num_particles = 1000
saving_interval = 10
logging_interval = 10
batch_size = 10
num_iterations = 500
num_particles = 100
proposal_scale_0 = 1
proposal_scale_t = 0.1

dataloader = train.get_synthetic_dataloader(
            forward_kin_2D.Initial(initial_loc, initial_scale),
            forward_kin_2D.Transition(true_transition_mult, transition_scale),
            forward_kin_2D.Emission(forward_kin_2D.FW_kin_2D, 
                                    L1_true, L2_true, emission_scale),
            num_timesteps, batch_size)

#%% not needed, these are tests.
num_iter_check=3
for epoch_iteration_idx, observations in enumerate(dataloader):
    if epoch_iteration_idx == num_iter_check:
                    break
                
print(observations[-1].size())
# should be [batch_size, observations]

sim_data = torch.cat([obs.unsqueeze(-1) for
                           obs in observations], dim=2)
print(sim_data.shape)


plt.plot(sim_data.detach().numpy()[0,4,:]);
plt.title('Batch 0: simulated time series', fontsize=14);
# ToDo: would like to get the video of the 2D arm. would also like to access latents

# inspect statistics.sample_from_prior
'''this function has access to latents and observations. '''
a,b = statistics.sample_from_prior(forward_kin_2D.Initial(initial_loc, initial_scale),
                              forward_kin_2D.Transition(true_transition_mult, transition_scale),
                              forward_kin_2D.Emission(forward_kin_2D.FW_kin_2D, 
                                 L1_init, L2_init, emission_scale), num_timesteps,
                                         batch_size)
print(a[-1].size()) # [batch_size, dim_latents]
print(b[-1].size()) # [batch_size, dim_obs]

#%%
training_stats = forward_kin_2D.TrainingStats(
                initial_loc, initial_scale, true_transition_mult,
                transition_scale,  L1_true, L2_true, emission_scale,
                num_timesteps, num_test_obs, test_inference_num_particles,
                saving_interval, logging_interval)

train.train(dataloader=dataloader,
            num_particles=num_particles,
            algorithm='aesmc',
            initial=forward_kin_2D.Initial(initial_loc, initial_scale),
            transition=forward_kin_2D.Transition(init_transition_mult,
                                        transition_scale),
            emission=forward_kin_2D.Emission(forward_kin_2D.FW_kin_2D, 
                                    L1_init, L2_init, emission_scale),
            proposal=forward_kin_2D.Proposal(proposal_scale_0,
                                    proposal_scale_t),
            num_epochs=1,
            num_iterations_per_epoch=num_iterations,
            callback=training_stats)

observations

torch.nn.Linear(

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:04:59 2020
Utils for smoothing the output of aesmc.inference.infer() using FFBS.
@author: danbiderman
"""
import torch
from aesmc import math
import numpy as np


def expand_normalize_weights(weight_tensor, k_realizations):
    '''seems to only be needed at step T. 
        Args: 
            weight_tensor: torch.tensor([batch_size, num_particles])
            k_realizations: integer that says how many smoothing realizations we need
        Returns:
             torch.tensor([k_realizations, batch_size, num_particles])
             repetitions across dim0, and all the particles (dim2) are summed to 1.
            '''
    expanded_weights = weight_tensor.unsqueeze(0). \
            expand(k_realizations, \
                   weight_tensor.shape[0], \
                   weight_tensor.shape[1]) 
    normalized_weights = math.exponentiate_and_normalize(
            expanded_weights.detach().cpu().numpy(), \
                dim=2) # dimension of particles
    
    return torch.tensor(normalized_weights)

def expand_chosen_particles(chosen_particles, num_particles):
    '''in: torch.Size([k_realizations, batch_size, dim_latents]), num_particles in PF
        out: torch.Size([k_realizations, batch_size, num_particles, dim_latents])
        where we repeat across dim = -2'''
    return chosen_particles.unsqueeze(-2).expand(
            chosen_particles.shape[0],
            chosen_particles.shape[1],
            num_particles, 
            chosen_particles.shape[2])

def weight_update(chosen_particles_tplus1, 
                  filtered_latents_t, 
                 log_weights_filtering_t,
                 transition_object):
    '''exapnd particles from step t+1, so that we can evaluate
    their log probability under the transition (which is expanded
    with num_particles). then add the log probability under transition
    to the log weights from filtering. exponentiate and normalize weights.'''
    latents_expanded = expand_chosen_particles(
        chosen_particles_tplus1, filtered_latents_t.shape[1]) # \tilde{x}_{t+1} expanded
    lp_trans = transition_object(
                [filtered_latents_t]).log_prob(
                latents_expanded) # f(\tilde{x}_{t+1}|x_t)
    w_t_given_tplus1 = log_weights_filtering_t + \
                        lp_trans 
    # broadcasting above, result is 
    # torch.Size([k_realizations, batch_size, num_particles])
    normalized_weights = torch.tensor(math.exponentiate_and_normalize(
                w_t_given_tplus1.detach().cpu().numpy(), dim=2))
    return normalized_weights

def sample_latents(normalized_weights, filtered_latents_t):
    '''normalized_weights: 
            torch.Size([k_realizations, batch_size, num_particles])
        filtered_latents_t:
            torch.Size([batch_size, num_particles, dim_latents])'''
    # ranomly choose indices
    sampled_indices = torch.distributions.Categorical(
                        normalized_weights).sample()
    # get associated latents
    latents = filtered_latents_t[np.tile(
            np.arange(normalized_weights.shape[1]),
            (normalized_weights.shape[0],1)), 
                sampled_indices, :]
    return sampled_indices, latents

def expand_distribution_obj(dist, k_realizations):
    '''assumes a multivariate normal distribution here
    maybe no need for that. I see that it gives the same results'''
    expanded_loc = dist.loc.unsqueeze(0).expand(k_realizations,
                                                dist.loc.shape[0],
                                                dist.loc.shape[1],
                                                dist.loc.shape[2])
    expanded_cov = dist.covariance_matrix.unsqueeze(0).expand(
        k_realizations, dist.covariance_matrix.shape[0],
        dist.covariance_matrix.shape[1], dist.covariance_matrix.shape[2],
        dist.covariance_matrix.shape[3])
    return torch.distributions.MultivariateNormal(
        loc=expanded_loc, covariance_matrix=expanded_cov)
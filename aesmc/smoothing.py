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

class smooth_result:
    def __init__(self, model, forward_inference_result, k_realizations):
        self.model = model
        self.forward_inference_result = forward_inference_result
        self.num_timesteps = len(self.forward_inference_result["original_latents"])
        self.k_realizations = k_realizations
        self.smoothing_result = {}  # dict of lists

    @staticmethod
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
            (normalized_weights.shape[0], 1)), sampled_indices, :]
        return sampled_indices, latents
    
    @staticmethod
    def expand_chosen_particles(chosen_particles, num_particles):
        '''in: torch.Size([k_realizations, batch_size, dim_latents]), num_particles in PF
            out: torch.Size([k_realizations, batch_size, num_particles, dim_latents])
            where we repeat across dim = -2'''
        return chosen_particles.unsqueeze(-2).expand(
                chosen_particles.shape[0],
                chosen_particles.shape[1],
                num_particles, 
                chosen_particles.shape[2])
    
    def expand_normalize_weights(self, weight_tensor):
        '''seems to only be needed at step T. 
            Args: 
                weight_tensor: torch.tensor([batch_size, num_particles])
                k_realizations: integer that says how many smoothing realizations we need
            Returns:
                 torch.tensor([k_realizations, batch_size, num_particles])
                 repetitions across dim0, and all the particles (dim2) are summed to 1.
                '''
        expanded_weights = weight_tensor.unsqueeze(0). \
                expand(self.k_realizations, \
                       weight_tensor.shape[0], \
                       weight_tensor.shape[1]) 
        normalized_weights = math.exponentiate_and_normalize(
                expanded_weights.detach().cpu().numpy(), \
                    dim=2) # dimension of particles

        return torch.tensor(normalized_weights)

    def weight_update(self, chosen_particles_tplus1, filtered_latents_t,
                      log_weights_filtering_t):
        '''exapnd particles from step t+1, so that we can evaluate
        their log probability under the transition (which is expanded
        with num_particles). then add the log probability under transition
        to the log weights from filtering. exponentiate and normalize weights.'''
        latents_expanded = self.expand_chosen_particles(
            chosen_particles_tplus1,
            filtered_latents_t.shape[1])  # \tilde{x}_{t+1} expanded
        lp_trans = self.model["transition"]([filtered_latents_t]).log_prob(
            latents_expanded)  # f(\tilde{x}_{t+1}|x_t)
        w_t_given_tplus1 = log_weights_filtering_t + \
                            lp_trans
        # broadcasting above, result is
        # torch.Size([k_realizations, batch_size, num_particles])
        normalized_weights = torch.tensor(
            math.exponentiate_and_normalize(
                w_t_given_tplus1.detach().cpu().numpy(), dim=2)) 
            #,device="cuda" if torch.cuda.is_available() else "cpu")
        return normalized_weights

    def run_backward_smoothing(self):

        self.smoothing_result["normalized_weights"] = []
        self.smoothing_result["original_latents"] = []
        self.smoothing_result["sampled_indices"] = []

        for t in range(self.num_timesteps - 1, -1, -1):
            if t == self.num_timesteps - 1:
                normalized_weights = self.expand_normalize_weights(
                    self.forward_inference_result["log_weights"][-1])  # Note index -1
                sampled_indices, latents = self.sample_latents(
                    normalized_weights,
                    self.forward_inference_result["original_latents"][-1])
            else:
                # compute weights
                normalized_weights = self.weight_update(
                    latents, self.forward_inference_result["original_latents"][t],
                    self.forward_inference_result["log_weights"][t])

                sampled_indices, latents = self.sample_latents(
                    normalized_weights,
                    self.forward_inference_result["original_latents"][t])

            # below these lists will all be flipped (in time)
            self.smoothing_result["normalized_weights"].append(
                normalized_weights)
            self.smoothing_result["original_latents"].append(latents)
            self.smoothing_result["sampled_indices"].append(sampled_indices)

        # flip order
        self.flip_smoothing_result()

    def flip_smoothing_result(self):
        for i in self.smoothing_result.keys():
            self.smoothing_result[i] = self.smoothing_result[i][::-1]

    def summarize(self):
        '''compute mean and variance for our k realizations'''
        self.smooth_traj = torch.cat([
            smooth.unsqueeze(-1) for smooth in self.smoothing_result["original_latents"]
        ],
                                     dim=3)
        smooth_mean = torch.mean(self.smooth_traj,
                                 dim=0).cpu().detach().numpy()  # over trajectories
        smooth_var = torch.var(self.smooth_traj, dim=0).cpu().detach().numpy()
        return smooth_mean, smooth_var


#%% old functions that work fine
''' all the functions below are functioning properly and can be called from a jupyter notebook'''

# def expand_normalize_weights(weight_tensor, k_realizations):
#     '''seems to only be needed at step T. 
#         Args: 
#             weight_tensor: torch.tensor([batch_size, num_particles])
#             k_realizations: integer that says how many smoothing realizations we need
#         Returns:
#              torch.tensor([k_realizations, batch_size, num_particles])
#              repetitions across dim0, and all the particles (dim2) are summed to 1.
#             '''
#     expanded_weights = weight_tensor.unsqueeze(0). \
#             expand(k_realizations, \
#                    weight_tensor.shape[0], \
#                    weight_tensor.shape[1]) 
#     normalized_weights = math.exponentiate_and_normalize(
#             expanded_weights.detach().cpu().numpy(), \
#                 dim=2) # dimension of particles
    
#     return torch.tensor(normalized_weights)

# def expand_chosen_particles(chosen_particles, num_particles):
#     '''in: torch.Size([k_realizations, batch_size, dim_latents]), num_particles in PF
#         out: torch.Size([k_realizations, batch_size, num_particles, dim_latents])
#         where we repeat across dim = -2'''
#     return chosen_particles.unsqueeze(-2).expand(
#             chosen_particles.shape[0],
#             chosen_particles.shape[1],
#             num_particles, 
#             chosen_particles.shape[2])

# def weight_update(chosen_particles_tplus1, 
#                   filtered_latents_t, 
#                  log_weights_filtering_t,
#                  transition_object):
#     '''exapnd particles from step t+1, so that we can evaluate
#     their log probability under the transition (which is expanded
#     with num_particles). then add the log probability under transition
#     to the log weights from filtering. exponentiate and normalize weights.'''
#     latents_expanded = expand_chosen_particles(
#         chosen_particles_tplus1, filtered_latents_t.shape[1]) # \tilde{x}_{t+1} expanded
#     lp_trans = transition_object(
#                 [filtered_latents_t]).log_prob(
#                 latents_expanded) # f(\tilde{x}_{t+1}|x_t)
#     w_t_given_tplus1 = log_weights_filtering_t + \
#                         lp_trans 
#     # broadcasting above, result is 
#     # torch.Size([k_realizations, batch_size, num_particles])
#     normalized_weights = torch.tensor(math.exponentiate_and_normalize(
#                 w_t_given_tplus1.detach().cpu().numpy(), dim=2))
#     return normalized_weights

# def sample_latents(normalized_weights, filtered_latents_t):
#     '''normalized_weights: 
#             torch.Size([k_realizations, batch_size, num_particles])
#         filtered_latents_t:
#             torch.Size([batch_size, num_particles, dim_latents])'''
#     # ranomly choose indices
#     sampled_indices = torch.distributions.Categorical(
#                         normalized_weights).sample()
#     # get associated latents
#     latents = filtered_latents_t[np.tile(
#             np.arange(normalized_weights.shape[1]),
#             (normalized_weights.shape[0],1)), 
#                 sampled_indices, :]
#     return sampled_indices, latents

# def expand_distribution_obj(dist, k_realizations):
#     '''assumes a multivariate normal distribution here
#     maybe no need for that. I see that it gives the same results'''
#     expanded_loc = dist.loc.unsqueeze(0).expand(k_realizations,
#                                                 dist.loc.shape[0],
#                                                 dist.loc.shape[1],
#                                                 dist.loc.shape[2])
#     expanded_cov = dist.covariance_matrix.unsqueeze(0).expand(
#         k_realizations, dist.covariance_matrix.shape[0],
#         dist.covariance_matrix.shape[1], dist.covariance_matrix.shape[2],
#         dist.covariance_matrix.shape[3])
#     return torch.distributions.MultivariateNormal(
#         loc=expanded_loc, covariance_matrix=expanded_cov)
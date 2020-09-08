#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:04:19 2020

@author: danbiderman
"""
import aesmc # note: imported from package, not local.
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Bootstrap_Proposal(nn.Module):
    """This proposal is proportional to the transition.
    at step zero, the proposal should be set to the initial distribution
    at step t, the proposal should be set to the transition
    Args: ToDo: update
    Returns:
        torch.distributions.Normal object. 
        at time=0, torch.tensor([batch_size, dim_latents]) 
        and at time!=0, torch.tensor([batch_size, num_particles, dim_latents])"""
    def __init__(self, initial_instance, transition_instance):
        super(Bootstrap_Proposal, self).__init__()
        self.initial = initial_instance
        self.transition = transition_instance
    
    def forward(self, previous_latents=None, time=None, observations=None):
        
        if time == 0: # initial
            return self.initial()
        else: # transition
            return self.transition.forward(previous_latents = previous_latents)

class Optimal_Proposal(nn.Module):
    '''Currently not supported for 3D. 
    this is an optimal proposal for a non-linear transition + linear
    emission model.'''
    def __init__(self, initial_instance, 
                 transition_instance,
                 emission_instance):
        super(Optimal_Proposal, self).__init__()
        # x_{t-1} = previous latents.
        # a(x_{t-1}) = current loc of transition.
        # all expressions below are tensors
        self.cov_0 = initial_instance.cov_mat
        self.precision_0 = torch.inverse(self.cov_0)
        self.mu_0 = initial_instance.loc 
        self.Q = transition_instance.diag_mat
        self.R = emission_instance.R
        self.C = emission_instance.C
        self.Q_inv = torch.inverse(self.Q)
        self.R_inv = torch.inverse(self.R)
        self.optimal_precision_t = self.Q_inv + torch.transpose(
                self.C, 0, 1).mm(self.R_inv.mm(self.C)) #  precision t>0
        self.optimal_cov_t = torch.inverse(self.optimal_precision_t) # covariance t>0
        self.optimal_precision_0 = self.precision_0 + \
                        torch.transpose(
                            self.C, 0, 1).mm(
                                self.R_inv.mm(self.C))
        self.optimal_cov_0 = torch.inverse(self.optimal_precision_0)
        self.dim_latents = self.cov_0.shape[0]
        self.transition = transition_instance
        
    def forward(self, previous_latents=None, time=None, observations=None):
        
        if time == 0:
            self.batch_size = observations[0].shape[0]
            
            optimal_loc = self.optimal_cov_0.matmul(
                self.precision_0.mm(self.mu_0.unsqueeze(-1)).expand(
                    self.batch_size,self.dim_latents,1) + \
                    torch.transpose(
                        self.C, 0, 1).matmul( # (6X4)
                            self.R_inv.matmul( # (4X4)
                                observations[0].unsqueeze(-1))) # (10X4X1)
                ).squeeze(-1)
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.MultivariateNormal(
                    loc=optimal_loc,
                    covariance_matrix=self.optimal_cov_0),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        
        else:
            ax_t_min_1 = self.transition.arm_model.forward(
                previous_latents)
            
            optimal_loc = self.optimal_cov_t.matmul(
                self.Q_inv.matmul(ax_t_min_1.unsqueeze(-1)) + \
                    torch.transpose(
                        self.C, 0, 1).matmul( # (6X4)
                            self.R_inv.matmul( # (4X4)
                                aesmc.state.expand_observation(
                                    observations[time], 
                                    previous_latents[-1].shape[1])
                                .unsqueeze(-1))) # (10X1000X4X1)
                ).squeeze(-1)
            
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.MultivariateNormal(
                    loc=optimal_loc,
                    covariance_matrix=self.optimal_cov_t),
                aesmc.state.BatchShapeMode.FULLY_EXPANDED)


class Learned_Proposal(nn.Module):
    """This Proposal uses a FF mapping between (1) observations[t] -> mu[t]
    and observations[t]} -> var[t].
    The weights and biases of each mapping could be learned. 
    Args:
        previous_latents: list of len num_timesteps, each entry is 
            torch.tensor([batch_size, num_particles, dim_latents])
        time: integer
        observations: list of len num_timesteps. each entry is a
        torch.tensor([batch_size, dim_observations]
    Returns:
        torch.distributions.Normal object. 
        at time=0, torch.tensor([batch_size, dim_latents]) 
        and at time!=0, torch.tensor([batch_size, num_particles, dim_latents])"""
    def __init__(self, initial_instance, 
                 transition_instance,
                 h_type = None,
                 num_hidden_units=None,
                 dim_obs=None,
                 num_timesteps=None):
        super(Learned_Proposal, self).__init__()
        # x_{t-1} = previous latents.
        # a(x_{t-1}) = current loc of transition.
        # all expressions below are tensors
        
        self.num_hidden_units = num_hidden_units # for FF nns
        # initial distribution parameters.
        self.sigma_squared_0 = torch.diag(initial_instance.cov_mat)
        self.mu_0 = initial_instance.loc 
        
        # transition dist. params.
        self.sigma_squared_t = torch.diag(transition_instance.diag_mat)
        self.transition = transition_instance

        self.dim_latents = self.sigma_squared_t.shape[0]
        self.dim_obs = dim_obs
        self.h_type = h_type
        print('our proposal is q=f*h, and type of h: %s' % self.h_type)
        
        if self.h_type == 'mlp':
            
            self.FF_mu = nn.Sequential(
                    nn.Linear(self.dim_obs, self.num_hidden_units),
                    nn.ReLU(),
                    nn.Linear(self.num_hidden_units,self.dim_latents)
                    ) # observations[t] -> mu[t]
            
            self.FF_var = nn.Sequential(
                    nn.Linear(self.dim_obs, self.num_hidden_units),
                    nn.ReLU(),
                    nn.Linear(self.num_hidden_units,self.dim_latents),
                    nn.Softplus()
                    ) # observations[t] -> sigma_squared[t]
            
            # input will be a double, so adapt model
            # see https://github.com/pytorch/pytorch/issues/2138
            self.FF_mu.double()
            self.FF_mu.to(device)
    
            self.FF_var.double()
            self.FF_var.to(device)
            
        elif self.h_type == 'tailored':
            self.num_timesteps = num_timesteps
            # rows: dims; cols: timesteps
            self.mu_mat = torch.nn.Parameter(torch.tensor(np.random.normal(loc=0.0, 
                                                        scale=1.0, 
                                                        size=(self.dim_latents, self.num_timesteps)),
                                       requires_grad=True, dtype = torch.double, device = device))
            self.sigma_mat = torch.nn.Parameter(torch.tensor(np.random.gamma(shape=0.5, 
                                                        scale=1.0, 
                                                        size=(self.dim_latents, self.num_timesteps)),
                                       requires_grad=True, dtype = torch.double, device = device))
            
            
        elif self.h_type == 'bi-LSTM':
            print('bi-LSTM')
        
        else:
            print('h type not supported')
        
    
    @staticmethod
    def get_sigma_squared_from_inverses(model_sigma_squared, sigma_squared_star):
        proposed_precision_vec = 1.0/model_sigma_squared + 1.0/sigma_squared_star
        return 1.0/proposed_precision_vec
    
    @staticmethod
    def get_mu(proposed_sigma_squared, model_sigma_squared, 
               model_mu, sigma_squared_star, mu_star):
        proposed_mu = proposed_sigma_squared*( \
            (1.0/model_sigma_squared)*model_mu + \
            (1.0/sigma_squared_star)*mu_star)
        return proposed_mu
    
    # @staticmethod
    # def expand_tensor(tensor, num_particles):
    #     batch_size = tensor.shape[0]
    #     dim = tensor.shape[1]
        

    def forward(self, previous_latents=None, time=None, observations=None):
        
        if self.h_type == 'mlp':
        
            sigma_squared_star = torch.clamp(self.FF_var(observations[time]),
                                             min = 0.01, max=30.0)
            mu_star = self.FF_mu(observations[time])
        
        elif self.h_type == 'tailored':
            '''take the current column of the param mat, and expand to (batch_size,dim_latents)'''
            sigma_squared_star = torch.clamp(self.sigma_mat[:, time], 
                                             min=0.01, max=100.0).expand(
                                                 observations[time].shape[0],-1)
            mu_star = torch.clamp(self.mu_mat[:, time], 
                                             min=-400.0, max=400.0).expand(
                                                 observations[time].shape[0],-1)
            
        elif self.h_type == 'bi-LSTM':
            if time == 0:
                print('run bi-LSTM for all time steps')
            print('now for t=0:T, take readouts to be mu and sigma.')
        
        else:
            print('h_type not supported.')
            
        assert(torch.sum(torch.isnan(mu_star))==0)
        assert(torch.sum(torch.isnan(sigma_squared_star))==0)
        
        if time == 0:
            self.batch_size = observations[0].shape[0]
            
            proposed_sigma_squared = self.get_sigma_squared_from_inverses(
                self.sigma_squared_0, 
                sigma_squared_star)
            
            proposed_mu = self.get_mu(
                proposed_sigma_squared, self.sigma_squared_0, 
                self.mu_0, sigma_squared_star, mu_star
                )

            return aesmc.state.set_batch_shape_mode(
                        torch.distributions.Normal(
                            loc=proposed_mu,
                            scale=torch.sqrt(proposed_sigma_squared)),
                        aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        else: # time > 0
            if time == 1:
                self.batch_size, self.num_particles, self.dim_latents = previous_latents[-1].shape
            
            fully_expanded_dims = [self.batch_size, 
                        self.num_particles, 
                        self.dim_latents]
            
            # transition sigma_squared
            sigma_squared_t_expanded = self.sigma_squared_t.unsqueeze(0).unsqueeze(0).expand(fully_expanded_dims)
            
            sigma_squared_star_expanded = sigma_squared_star.unsqueeze(1).expand(
                fully_expanded_dims)
            
            mu_star_expanded = mu_star.unsqueeze(1).expand(
                fully_expanded_dims)
            
            proposed_sigma_squared = self.get_sigma_squared_from_inverses(
                sigma_squared_t_expanded, 
                sigma_squared_star_expanded)
            
            # mu_t is the deterministic forward dynamics
            mu_t = self.transition.arm_model.forward(
                previous_latents)
                        
            proposed_mu = self.get_mu(
                proposed_sigma_squared, sigma_squared_t_expanded, 
                mu_t, sigma_squared_star_expanded, mu_star_expanded
                )
            
            
            return aesmc.state.set_batch_shape_mode(
                        torch.distributions.Normal(
                            loc=proposed_mu,
                            scale=torch.sqrt(proposed_sigma_squared)),
                        aesmc.state.BatchShapeMode.FULLY_EXPANDED)
                  

## Bellow is working perfectly fine. we just want to add more flexibility. 
# class Learned_Proposal(nn.Module):
#     """This Proposal uses a FF mapping between (1) observations[t] -> mu[t]
#     and observations[t]} -> var[t].
#     The weights and biases of each mapping could be learned. 
#     Args:
#         previous_latents: list of len num_timesteps, each entry is 
#             torch.tensor([batch_size, num_particles, dim_latents])
#         time: integer
#         observations: list of len num_timesteps. each entry is a
#         torch.tensor([batch_size, dim_observations]
#     Returns:
#         torch.distributions.Normal object. 
#         at time=0, torch.tensor([batch_size, dim_latents]) 
#         and at time!=0, torch.tensor([batch_size, num_particles, dim_latents])"""
#     def __init__(self, initial_instance, 
#                  transition_instance, 
#                  num_hidden_units,
#                  dim_obs):
#         super(Learned_Proposal, self).__init__()
#         # x_{t-1} = previous latents.
#         # a(x_{t-1}) = current loc of transition.
#         # all expressions below are tensors
        
#         self.num_hidden_units = num_hidden_units # for FF nns
#         # initial distribution parameters.
#         self.sigma_squared_0 = torch.diag(initial_instance.cov_mat)
#         self.mu_0 = initial_instance.loc 
        
#         # transition dist. params.
#         self.sigma_squared_t = torch.diag(transition_instance.diag_mat)
#         self.transition = transition_instance

#         self.dim_latents = self.sigma_squared_t.shape[0]
#         self.dim_obs = dim_obs
        
#         self.FF_mu = nn.Sequential(
#                 nn.Linear(self.dim_obs, self.num_hidden_units),
#                 nn.ReLU(),
#                 nn.Linear(self.num_hidden_units,self.dim_latents)
#                 ) # observations[t] -> mu[t]
        
#         self.FF_var = nn.Sequential(
#                 nn.Linear(self.dim_obs, self.num_hidden_units),
#                 nn.ReLU(),
#                 nn.Linear(self.num_hidden_units,self.dim_latents),
#                 nn.Softplus()
#                 ) # observations[t] -> sigma_squared[t]
        
#         # input will be a double, so adapt model
#         # see https://github.com/pytorch/pytorch/issues/2138
#         self.FF_mu.double()
#         self.FF_mu.to(device)

#         self.FF_var.double()
#         self.FF_var.to(device)
        
    
#     @staticmethod
#     def get_sigma_squared_from_inverses(model_sigma_squared, sigma_squared_star):
#         proposed_precision_vec = 1.0/model_sigma_squared + 1.0/sigma_squared_star
#         return 1.0/proposed_precision_vec
    
#     @staticmethod
#     def get_mu(proposed_sigma_squared, model_sigma_squared, 
#                model_mu, sigma_squared_star, mu_star):
#         proposed_mu = proposed_sigma_squared*( \
#             (1.0/model_sigma_squared)*model_mu + \
#             (1.0/sigma_squared_star)*mu_star)
#         return proposed_mu
    
#     # @staticmethod
#     # def expand_tensor(tensor, num_particles):
#     #     batch_size = tensor.shape[0]
#     #     dim = tensor.shape[1]
        

#     def forward(self, previous_latents=None, time=None, observations=None):
        
#         sigma_squared_star = torch.clamp(self.FF_var(observations[time]),
#                                          min = 0.01, max=30.0)
#         mu_star = self.FF_mu(observations[time])
            
#         assert(torch.sum(torch.isnan(mu_star))==0)
#         assert(torch.sum(torch.isnan(sigma_squared_star))==0)
        
#         if time == 0:
#             self.batch_size = observations[0].shape[0]
            
#             proposed_sigma_squared = self.get_sigma_squared_from_inverses(
#                 self.sigma_squared_0, 
#                 sigma_squared_star)
            
#             proposed_mu = self.get_mu(
#                 proposed_sigma_squared, self.sigma_squared_0, 
#                 self.mu_0, sigma_squared_star, mu_star
#                 )

#             return aesmc.state.set_batch_shape_mode(
#                         torch.distributions.Normal(
#                             loc=proposed_mu,
#                             scale=torch.sqrt(proposed_sigma_squared)),
#                         aesmc.state.BatchShapeMode.BATCH_EXPANDED)
#         else: # time > 0
#             if time == 1:
#                 self.batch_size, self.num_particles, self.dim_latents = previous_latents[-1].shape
            
#             fully_expanded_dims = [self.batch_size, 
#                         self.num_particles, 
#                         self.dim_latents]
            
#             # transition sigma_squared
#             sigma_squared_t_expanded = self.sigma_squared_t.unsqueeze(0).unsqueeze(0).expand(fully_expanded_dims)
            
#             sigma_squared_star_expanded = sigma_squared_star.unsqueeze(1).expand(
#                 fully_expanded_dims)
            
#             mu_star_expanded = mu_star.unsqueeze(1).expand(
#                 fully_expanded_dims)
            
#             proposed_sigma_squared = self.get_sigma_squared_from_inverses(
#                 sigma_squared_t_expanded, 
#                 sigma_squared_star_expanded)
            
#             # mu_t is the deterministic forward dynamics
#             mu_t = self.transition.arm_model.forward(
#                 previous_latents)
                        
#             proposed_mu = self.get_mu(
#                 proposed_sigma_squared, sigma_squared_t_expanded, 
#                 mu_t, sigma_squared_star_expanded, mu_star_expanded
#                 )
            
            
#             return aesmc.state.set_batch_shape_mode(
#                         torch.distributions.Normal(
#                             loc=proposed_mu,
#                             scale=torch.sqrt(proposed_sigma_squared)),
#                         aesmc.state.BatchShapeMode.FULLY_EXPANDED)
                  

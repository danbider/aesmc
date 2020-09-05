#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:48:32 2020

@author: danbiderman
"""
import torch
import aesmc # note: imported from package
import aesmc.statistics as statistics
import numpy as np

def summarize_posterior(inference_result_list):
    '''compute empirical mean and variance using the final particle weights.'''
    
    # compute mean and variance of the latents
    smc_smoothed_state_means = []
    smc_smoothed_state_variances = []
    for latent in inference_result_list['latents']:
        smc_smoothed_state_means.append(statistics.empirical_mean(
            latent, inference_result_list['log_weight']
        ))
        smc_smoothed_state_variances.append(statistics.empirical_variance(
            latent, inference_result_list['log_weight']
        ))
    
    # compute effective sample size per timestep
    ESS = []
    for weight_set in inference_result_list["log_weights"]:
        ESS.append(statistics.ess(weight_set)) 

    # transform into a numpy array of shape batch_size X dim_latents X num_timesteps
    smooth_mean =  torch.cat([mean.unsqueeze(-1) for
                               mean in smc_smoothed_state_means], dim=2).detach().numpy()
    smooth_var = torch.cat([var.unsqueeze(-1) for
                               var in smc_smoothed_state_variances], dim=2).detach().numpy()
    ESS_cat =  torch.cat([ESS_i.unsqueeze(-1) for
                           ESS_i in ESS],
                         dim=1).detach().numpy()
    
    posterior_summary = {}
    posterior_summary["smooth_mean"] = smooth_mean
    posterior_summary["smooth_var"] = smooth_var
    posterior_summary["ESS"] = ESS_cat
    
    return posterior_summary

def summarize_independent_smc_samplers(post_mean, post_var,
                                       logw=None,
                                       method='uniform',
                                       remove_indices=None):
    if remove_indices is not None:
        """helpful in the case that one estimate dominates the rest."""
        post_mean = np.delete(
           post_mean, remove_indices, axis=0)
        post_var = np.delete(
            post_var, remove_indices, axis=0)
        if logw is not None:
            logw = np.delete(logw, remove_indices)
    print('applying %s average of %i SMC samplers.' %
          (method, post_mean.shape[0]))
    if method == 'uniform':
        mean = np.mean(post_mean, axis=0)
        var = np.mean(post_var, axis=0)
    elif method == 'weighted':
        from scipy.special import logsumexp
        w = np.exp(logw - logsumexp(logw))
        mean = np.average(post_mean, axis=0, weights=w)
        var = np.average(post_var, axis=0, weights=w)
    else:
        print('method not supported. try weighted or uniform')
    return mean, var

def send_inference_result_to_cpu(inference_result):
    if torch.cuda.is_available():
        keys = inference_result.keys()
        for key in keys:
            #print(key)
            if type(inference_result[key]) is torch.Tensor:
                inference_result[key] = inference_result[key].cpu()
            elif type(inference_result[key]) is list:
                if len(inference_result[key]) > 0:
                    for i in range(len(inference_result[key])):
                        inference_result[key][i] = inference_result[key][i].cpu()
    return inference_result
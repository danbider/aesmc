#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:46:30 2020

@author: danbiderman
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

def video_sim_plane_2D(x_obs, y_obs, x_star, y_star, save_name, title, n_frames, lim_abs):
    
    from matplotlib.animation import FuncAnimation

    fig = plt.figure()
    ax = plt.axes(xlim=(-lim_abs, lim_abs), ylim=(-lim_abs, lim_abs))
    plt.xlabel("x", fontsize = 16)
    plt.ylabel("y", fontsize = 16)
    line_noise, = ax.plot([], [],'r*',markersize=12, label = "observed")
    line_hat, = ax.plot([], [], 'ko-', lw=3, markersize =6, label = "predicted")
    plt.legend(loc = "lower left")
    plt.title(title, fontsize = 16)
    def init():
        line_noise.set_data([], [])
        line_hat.set_data([],[])
        return line_noise, line_hat
    def animate(i):
        line_noise.set_data(x_obs[i,:], y_obs[i,:])
        line_hat.set_data(x_star[i,:], y_star[i,:])
        return line_noise, line_hat

    anim = FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=50, blit=True)

    #anim.save("recover.mp4")
    anim.save(save_name + '.gif', writer='pillow') # works
#from matplotlib.animation import FFMpegWriter
#writer = FFMpegWriter() # fps=15, bitrate=3600, metadata=dict(artist='Me')
#save_name = 'recover' + '.mp4'
#anim.save(save_name, writer=writer)
    
def coords_from_params_mat(q1, q2, a1, a2):
    '''all inputs are vectors, outputs are mats
    nrows = length of input vecs i.e., timepoints
    cols correspond to joints'''
    x = np.zeros((len(q1),3))
    y = np.zeros((len(q1),3))
    x[:,0] = np.zeros(len(q1)) 
    x[:,1] = a1*np.cos(q1)
    x[:,2] = a2*np.cos(q1+q2) + a1*np.cos(q1)
    y[:,0] = np.zeros(len(q1))
    y[:,1] = a1*np.sin(q1)
    y[:,2] = a2*np.sin(q1+q2) + a1*np.sin(q1)
    return x,y

def plot_posterior_trace(post_mean = None, post_var = None, data_vec = None, 
                         alpha = 1.0, plot_legend=True, 
                         plot_uncertainty=False, plot_true_data=False,legends_list = None, 
                         legend_loc = "lower left", title = None, 
                         xlabel = 'Time', ylabel = 'Amplitude', fig_fullname = None, 
                         ax = None):
    '''a function that plots a time series signal, optionally a variance est,
    and with ground-truth or data signal. 
    Args: 
        post_mean = a 1D time series, numpy.array of shape (num_timesteps,)
        post_var = a 1D time series of some empirical var, or std, numpy.array of shape (num_timesteps,)
        data_vec = another 1D time series, numpy.array of shape (num_timesteps,)
        alpha = float in [0,1], for the optional fill_between function for post vars
    Returns: a figure on ax
    '''
    ax = ax or plt.gca()
    
    p1, = ax.plot(post_mean, color = "black", linestyle = "dashdot", zorder=2)
    if plot_uncertainty:
        p2 = ax.fill_between(np.arange(len(post_mean)), post_mean - post_var,
                         post_mean + post_var, color='gray', alpha=alpha, zorder=3)
    if plot_true_data:
        p3, = ax.plot(data_vec, color = 'red', zorder=1)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #plt.yticks(ticks = [0, .5, 1, 1.5], fontsize=14)
    #ax.set_yticks(ticks = input_ticks)
    #ax.tick_params(axis='y', labelsize= 12)
    #ax.set_yticklabels(input_labels)
    #ax.set_xticks([0,150, 300, 450])
    #ax.tick_params(axis='x', labelsize= 12)

    if plot_uncertainty:
        model_obj = (p1, p2)
        #legends = ["Arm Model (90% CI)", "DeepLabCut"]

    else:
        model_obj = (p1)
        #legends = ["Arm Model", "DeepLabCut"]
    if plot_legend == True:
        if plot_true_data == False:
            legends_list = [legends_list[0]]
            ax.legend([model_obj], legends_list, loc = legend_loc, 
                    frameon=False)
        else:
            ax.legend([model_obj, p3], legends_list, loc = legend_loc, 
                        frameon=False)
    return ax

# functions below were used for PyParticleEst and COSYNE submission
def center_3D_coords(sliced, use_mean = True):
    if use_mean:
        x_offset = np.nanmean(sliced[:,0])
        y_offset = np.nanmean(sliced[:,1])
        z_offset = np.nanmean(sliced[:,2])
        temp = [x_offset,y_offset,z_offset]
        temp_tile = np.tile(temp,3)
    else: 
        x_offset = sliced[:,0]
        y_offset = sliced[:,1]
        z_offset = sliced[:,2]
        temp = np.array([x_offset,y_offset,z_offset]).T
        temp_tile = np.tile(temp,(1,3))
    
    data_centered = np.copy(sliced)
    data_centered = data_centered - temp_tile
    
    if use_mean == False:
        assert((data_centered[:,0] == 0).all())
        assert((data_centered[:,1] == 0).all())
        assert((data_centered[:,2] == 0).all())
    
    return data_centered, (x_offset, y_offset, z_offset)

def uncenter_3D_coords(sliced, x_offset, y_offset, z_offset):
    
    temp = [x_offset,y_offset,z_offset]
    temp_tile = np.tile(temp,3)
    
    data_non_centered = np.copy(sliced)
    data_non_centered = data_non_centered + temp_tile
    return data_non_centered

def compute_2D_norms(data_mat):
    a1_pre_norm = data_mat[:,[2,3]]-data_mat[:,[0,1]]
    a2_pre_norm = data_mat[:,[4,5]]-data_mat[:,[2,3]]
    norm_1 = np.linalg.norm(a1_pre_norm, axis=1)
    norm_2 = np.linalg.norm(a2_pre_norm, axis=1)
    return norm_1, norm_2

def compute_3D_norms(data_mat):
    a1_pre_norm = data_mat[:,[3,4,5]]-data_mat[:,[0,1,2]]
    a2_pre_norm = data_mat[:,[6,7,8]]-data_mat[:,[3,4,5]]
    norm_1 = np.linalg.norm(a1_pre_norm, axis=1)
    norm_2 = np.linalg.norm(a2_pre_norm, axis=1)
    return norm_1, norm_2

def plot_empirical_norms_hist(norm1, norm2, bins, filename):
    plt.subplot(121)
    plt.title(r'$\hat{L}_1$')
    plt.xlabel('Pixels')
    plt.ylabel('Count')
    plt.title(r'$\hat{L}_1\;$' + 'via' + r'$\;||\mathbf{y}_{Elbow}-\mathbf{y}_{Shoulder}||_2^2$')

    list_values = plt.hist(norm1, color = 'gray', bins=bins)
    plt.vlines(x=np.median(norm1), 
               ymin=0, ymax=np.max(list_values[0]), 
               label='median')
    plt.vlines(x=np.mean(norm1), 
               ymin=0, ymax=np.max(list_values[0]), 
               linestyle = 'dashed',
               label='mean')
    plt.legend()
    plt.subplot(122)
    plt.title(r'$\hat{L}_2\;$' + 'via' + r'$\;||\mathbf{y}_{EE}-\mathbf{y}_{Elbow}||_2^2$')
    plt.xlabel('Pixels')
    plt.ylabel('Count')
    list_values = plt.hist(norm2, color = 'gray', bins=bins)
    plt.vlines(x=np.median(norm2), 
               ymin=0, ymax=np.max(list_values[0]), 
               label='median')
    plt.vlines(x=np.mean(norm2), 
               ymin=0, ymax=np.max(list_values[0]),
               linestyle = 'dashed',
               label='mean')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename + '.png')
    plt.show()

import pickle
# pickle utils
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:  # note rb and not wb
        return pickle.load(input)

def plot_simulated_data(lat_data_np, 
                        sim_data_np, 
                        ind_in_batch,
                        label_dict,
                       fig_full_path):
    '''plot a subplot for a simulated dataset. on the left
    plot states, on the right plot observations. 
    Arguments:
        lat_data_np: numpy array shape (batch_size, dim_latents, num_timesteps)
        sim_data_np: numpy array shape (batch_size, dim_obs, num_timesteps)
        ind_in_batch: integer, index of a single batch we want to visualize
        label_dict: dict with keys ["state"] and ["obs"], each including a list of labels
        fig_full_path: string, where to save.
    '''
    plt.subplot(121)
    plt.plot(lat_data_np[ind_in_batch, :, :].T)
    plt.title('simulated states')
    plt.ylabel('variable values')
    plt.xlabel('time / dt')
    plt.legend(label_dict["state"])
    plt.subplot(122)
    plt.plot(sim_data_np[ind_in_batch, :, :].T)
    plt.title('simulated observations')
    plt.ylabel('variable values')
    plt.xlabel('time / dt')
    plt.legend(label_dict["obs"])
    plt.tight_layout()
    plt.savefig(fig_full_path)
    
def plot_3d_points(coord_list_of_dicts,
                   lims_dict,
                   index,
                   color_list,
                   ax):
    '''plot a single frame
    ToDo: specify plotting patams, allow for multiple inputs'''
    
    assert(len(color_list)==len(coord_list_of_dicts))

    ax.set_xlim3d(lims_dict["x"])
    ax.set_ylim3d(lims_dict["y"])
    ax.set_zlim3d(lims_dict["z"])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    for i in range(len(coord_list_of_dicts)):
        ax.plot(coord_list_of_dicts[i]["x_coords"][index,:], \
                coord_list_of_dicts[i]["y_coords"][index,:], \
                coord_list_of_dicts[i]["z_coords"][index,:],
                linestyle = 'None', 
                marker = 'o', 
                color = color_list[i], markersize=4, label = None)

def arr3d_to_dict(arr):
    'convert n_frames by 9 array to a dict with x,y,z coords'
    dict_3d = {}
    dict_3d["x_coords"] = arr[:,[0, 3, 6]].T
    dict_3d["y_coords"] = arr[:,[1, 4, 7]].T
    dict_3d["z_coords"] = arr[:,[2, 5, 8]].T
    return dict_3d

from scipy.interpolate import CubicSpline

class interpolate_dataset:
    
    def __init__(self, orig_dataset, orig_sampling_rate):
        self.orig_dataset = orig_dataset
        self.orig_sampling_rate = orig_sampling_rate
        self.num_timesteps_orig = orig_dataset.shape[-1] # assuming dataset is (dim_obs, num_timesteps)
        self.orig_x = np.arange(start=0.0,
              stop=self.num_timesteps_orig * 1.0 / self.orig_sampling_rate,
              step=1.0 / self.orig_sampling_rate)
    
    def interpolate_cubic(self):
        return CubicSpline(self.orig_x, self.orig_dataset.T)
    
    def interpolate_evaluate(self, num_timesteps_desired, desired_sampling_rate):
        '''the desired sampling rate should be a multiple of the original
        sampling rate so that we can later pick those time points where it'''
        assert(desired_sampling_rate%self.orig_sampling_rate==0)
        self.new_x = np.arange(start=0.0,
              stop=num_timesteps_desired * 1.0 / desired_sampling_rate,
              step=1.0 / desired_sampling_rate)
        # https://stackoverflow.com/questions/51744613/numpy-setdiff1d-with-tolerance-comparing-a-numpy-array-to-another-and-saving-o/51747164#51747164
        self.timesteps_to_take = ~(np.abs(np.subtract.outer(self.new_x,self.orig_x)) > 0.00001).all(1)#np.isin(self.new_x, self.orig_x) # bool
        cubic_spline = self.interpolate_cubic()
        return np.copy(cubic_spline(self.new_x).T)
    
def arr_to_list_of_tensors(dataset_arr, batch_size):
    '''dataset_arr is dim_obs by num_timesteps
    batch_size assumes that we repeat across batches for multiple independent SMC'''
    observations = []
    for i in range(dataset_arr.shape[1]):
        temp_tens = torch.tensor(dataset_arr[:, i]). \
            to("cuda" if torch.cuda.is_available() else "cpu")
        if batch_size>1:
            temp_tens.unsqueeze(0).expand(
                batch_size, dataset_arr.shape[0])
        
        observations.append(temp_tens)
    return observations

def optimize_angles_single_frame(batch_size,
                                 FW_kin_func,
                                 num_angles,
                                 observation,
                                 optim_steps=20000,
                                 replicated_batch=False,
                                 init_angles_from_prev=None):

    if replicated_batch:
        batch_size = 1
        obs_to_fit = observation[0, :].unsqueeze(0)
    else:
        obs_to_fit = observation

    if init_angles_from_prev is not None:
        assert (init_angles_from_prev.shape == (batch_size, 1, num_angles))
        angles_init = torch.tensor(
            init_angles_from_prev, requires_grad=True).to(
                "cuda" if torch.cuda.is_available() else "cpu")
    else:
        angles_init = torch.tensor(
            np.random.vonmises(mu=0, kappa=8, size=(batch_size, 1, num_angles)),
            requires_grad=True, device = "cuda" if torch.cuda.is_available() else "cpu")
        #.to("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam([angles_init], weight_decay=1e-5)
    loss_list = []
    for i in range(optim_steps):
        optimizer.zero_grad()
        loss = torch.mean(
            torch.sum((FW_kin_func(torch.clamp(angles_init, min=-np.pi, max = np.pi)).squeeze(1) - obs_to_fit)**2,
                      dim=1))  # for all of the batch elements
        loss_list.append(loss)
        loss.backward()
        optimizer.step()

    return angles_init.cpu().detach().numpy(), loss_list
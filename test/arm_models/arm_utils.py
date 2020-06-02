#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:46:30 2020

@author: danbiderman
"""
import numpy as np
import matplotlib.pyplot as plt

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
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
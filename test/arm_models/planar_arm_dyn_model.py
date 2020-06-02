#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:05:00 2020

@author: danbiderman
"""

import copy
import aesmc # note: imported from package, not local.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarArmDyn(nn.Module): # 06/01: added inheritence nn.Module
    ''''the deterministics dynamics of the arm model. 
        Computes the mean for transition(t)
        and could also appear in the proposal (either optimal or BPF)'''
    def __init__(self, dt, inits_dict, 
                 g, include_gravity_fictitious = True, 
                 transform_torques=False, learn_static=False):
        super(PlanarArmDyn, self).__init__()
        self.dt = dt
        # self.M1 = inits_dict["M1"] # upper arm mass
        # self.M2 = inits_dict["M2"] # forearm mass
        self.L1 = inits_dict["L1"] # upper arm length
        self.L2 = inits_dict["L2"] # forearm length
        self.M1 = nn.Parameter(torch.Tensor([inits_dict["M1"]]).squeeze(), 
                               requires_grad = learn_static) # upper arm mass
        self.M2 = nn.Parameter(torch.Tensor([inits_dict["M2"]]).squeeze(), 
                               requires_grad = learn_static) # forearm mass
        self.g = g # gravity constant
        self.dim_latents = 6
        self.include_gravity_fictitious = include_gravity_fictitious
        self.transform_torques = transform_torques
        self.A = torch.tensor((-0.25)*np.eye(2), dtype = torch.double) # add it as a parameter and input to init.
        # ToDo: could have instead of A, dim_intermediate = 10
        #  self.lin_t = nn.Sequential(
        #               nn.Linear(2, dim_intermediate), 
        #               nn.ReLU(),
        #               nn.Linear(dim_intermediate, 2)) #

        
    def D(self, t2):
        '''computes inertia tensor from static parameters and the current angle t2.
        in a deterministic world without batches and particles, this is a 2X2 matrix.
        Args: 
            t2: current elbow angle. torch.Tensor(batch_size * num_particles)
        Returns:
            torch.tensor [batch_size*num_particles, 2, 2]. These dimensions are
            important [large_batch_size, [squared mat]] 
            because they allow us to later invert D'''
        D_tensor = torch.zeros(t2.shape[0], 2, 2, dtype = torch.double)
        D_tensor[:,0,0] = self.L1**2*self.M1/3 + self.M2*(3*self.L1**2 + 
                            3*self.L1*self.L2*torch.cos(t2) + self.L2**2)/3
        D_tensor[:,0,1] = self.L2*self.M2*(3*self.L1*torch.cos(t2) + 2*self.L2)/6
        D_tensor[:,1,0] = self.L2*self.M2*(3*self.L1*torch.cos(t2) + 2*self.L2)/6
        D_tensor[:,1,1] = self.L2**2*self.M2/3*torch.ones_like(t2)
        
        return D_tensor
    
    def h(self, dt1, dt2, t2):
        '''Coriolis and centripetal vector, a length-2 column vector.
        function of static params and instantaneus angular velocities and elbow angle
        Args: 
            dt1: current shoulder angular velocity. torch.Tensor(batch_size * num_particles)
            dt2: current elbow angular velocity. torch.Tensor(batch_size * num_particles)
            t2: current elbow angle. torch.Tensor(batch_size * num_particles)
        Returns:
            torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
            dot product later; should be a length-2 column vector.
            '''
        assert(dt1.shape[0] == dt2.shape[0] & dt2.shape[0] == t2.shape[0])
        # note: we are always assuming batched processing.
        h_vec_tensor = torch.zeros(t2.shape[0], 2, 1) # output shape
        h_vec_tensor[:,0,0] = -self.L1*self.L2*self.M2*dt2*(2*dt1 + dt2)*torch.sin(t2)/2
        h_vec_tensor[:,1,0] = self.L1*self.L2*self.M2*dt1**2*torch.sin(t2)/2
    
        return h_vec_tensor
    
    def c(self, t1, t2):
        '''gravity vector, a length-2 column vector. 
        function of all static parameters, current configuration of both angles 
        and gravity constant.
        Args: 
            t1: current shoulder angle. torch.Tensor(batch_size * num_particles)
            t2: current elbow angle. torch.Tensor(batch_size * num_particles)
        Returns:
            torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
            dot product later; should be a length-2 column vector.
            '''
        assert(t1.shape[0] == t2.shape[0])
        # note: we are always assuming batched processing.
    
        c_vec_tensor = torch.zeros(t2.shape[0], 2, 1) # output shape
        c_vec_tensor[:,0,0] = self.L1*self.M1*self.g*torch.cos(t1)/2 + \
                self.M2*self.g*(2*self.L1*torch.cos(t1) + \
                self.L2*torch.cos(t1 + t2))/2
        c_vec_tensor[:,1,0] = self.L2*self.M2*self.g*torch.cos(t1 + t2)/2
        
        return c_vec_tensor
    
    @staticmethod
    def Newton_2nd(torque_vec_tens, 
                   D_mat_tens, h_vec_tens, c_vec_tens):
        # ToDo: think whether we want to log accel.
        '''compute instantaneous angular acceleration, a length-2 column vector, 
        according to Newton's second law: 
            force = mass*acceleration, 
        which in the angular setting (and in 2D), becomes: 
            torque (2X1) = intertia_tens (2X2) * angular_accel (2X1)
        Here, we are multiplying both sides of the equation by 
        intertia_tens**(-1) to remain with an expression for angular acceleration.
        We are also taking into account "fictitious forces" coriolis/centripetal and gravity
        forces. 
        The function executes three operations:
            (1) invert the intertia tensor, 
            (2) subtract "fictitious forces" from torque vector
            (3) compute the dot product between (1) and (2)
        
        Args: 
            torque_vec_tens: latent forces from the last time point, 
                torch.tensor [batch_size*num_particles, 2, 1].
                ToDo: make sure it is viewed that way.
            
            D_mat_tens: output of self.D, inertia tensor that is computed from
                static parameters and angles.
                torch.tensor [batch_size*num_particles, 2, 2]. These dimensions are
                important [batch_size * num_particles, [squared mat]] 
                because they allow us to later invert D
            
            h_vec_tens: output of self.h_vec, Coriolis and centripetal vector, 
                computed from static params and 
                instantaneus angular velocities and elbow angle.
                torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
                dot product later; should be a length-2 column vector.
            
            c_vec_tens: output of self.c_vec, gravity vector.
                computed from static parameters, current configuration of both angles 
                and gravity constant.
                torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
                dot product later; should be a length-2 column vector.
            
        Returns:
            torch.Tensor([batch_size * num_particles, 2]). Should be a length-2 column vector
            if we have two angles -- matching the angular velocity and angle vectors.
            TODO: make sure that this tensor is logged when this function is called.
            potentially use wrapper that logs it inside emission. 
            in general, consider logging h_vec and c_vec
            '''
        D_inv_mat_tens = torch.inverse(D_mat_tens)
        brackets = torque_vec_tens - h_vec_tens - c_vec_tens 
        inst_accel = D_inv_mat_tens.matmul(brackets)
        # want the output of this to be 
        # size [batch_size, num_part, 2]. we view it differently outside
        return inst_accel # inst_accel.squeeze()
    
    def forward(self, previous_latents=None):
        '''input: previous_latents 
            torch.Tensor([batch_size, num_particles, dim_latents])
            out: mean_fully_expanded 
            torch.Tensor([batch_size, num_particles, dim_latents]) 
            deterministic argument for either transition/proposal'''
        
        batch_size = previous_latents[-1].shape[0]
        num_particles = previous_latents[-1].shape[1]
        
        # compute a(x_{k-1}) of shape (batch_size*num_particles,6)
        ax = torch.zeros([batch_size, 
                          num_particles, 
                          self.dim_latents], dtype = torch.double) # save room

        ax[:,:,2] = previous_latents[-1][:,:,4] # \dot{theta}_1
        ax[:,:,3] = previous_latents[-1][:,:,5] # \dot{theta}_2
        
        # fill in the last two entries with Newton's second law

        inert_tens = self.D(previous_latents[-1][:,:,3].\
                                  contiguous().view(batch_size*num_particles)) # 4th element in state
        torque_vec = previous_latents[-1][:,:,:2].\
            contiguous().view(batch_size*num_particles,2,1) # first two elements in state vec
        
        if self.transform_torques:
            """instead of ax[:,:,:2] = 0, we insert \dot{\tau} = A\tau"""
            ax[:,:,:2] = self.A.unsqueeze(0).expand(
                            batch_size*num_particles,2,2).matmul(
                                torque_vec).view(
                                    batch_size, num_particles, 2)                 
            
        if self.include_gravity_fictitious:
            h_vec = self.h(dt1 = previous_latents[-1][:,:,4].\
                                      contiguous().view(batch_size*num_particles), 
                           dt2 = previous_latents[-1][:,:,5].\
                                      contiguous().view(batch_size*num_particles), 
                            t2 = previous_latents[-1][:,:,3].\
                                      contiguous().view(batch_size*num_particles))
            c_vec = self.c(t1 = previous_latents[-1][:,:,2].\
                                      contiguous().view(batch_size*num_particles),
                           t2 = previous_latents[-1][:,:,3].\
                                      contiguous().view(batch_size*num_particles))
        else: # zero gravity and fictitious forces
            # a version that worked
            c_vec = torch.zeros_like(torque_vec)
            h_vec = torch.zeros_like(torque_vec)
        
        # compute accel using Netwon's second law
        accel = self.Newton_2nd(torque_vec, 
                                      inert_tens, h_vec, c_vec)
        
        ax[:,:,4:] = accel.view(batch_size, num_particles, 2)

        # deterministic first order Euler integration
        mean_fully_expanded = previous_latents[-1] + self.dt * ax
        
        return mean_fully_expanded
        
        
class Initial: # could be made specific for each variable, or learned 
    # ToDo: if angles and velocities are observed, use them to initialize
    # each batch properly?
    '''distribution for latents at t=0, i.i.d draws from normal(loc,scale)'''
    def __init__(self, loc, cov_mat):
        self.loc = torch.tensor(loc, dtype = torch.double)
        self.cov_mat = torch.tensor(cov_mat, dtype = torch.double)

    def __call__(self): 
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.MultivariateNormal(
            self.loc, self.cov_mat), aesmc.state.BatchShapeMode.NOT_EXPANDED)

# class Transition(nn.Module): 
#     '''x \in R^6 := [tau_1, tau_2, theta_1, theta_2, 
#     \dot{theta}_1, \dot{theta}_2]^T'''
#     def __init__(self, dt, inits_dict, scale_force, scale_aux, g):
#         super(Transition, self).__init__()
#         self.dt = dt
#         self.scale_force = scale_force 
#         self.scale_aux = scale_aux # scale aux should be smaller than dt
#         self.diag_mat = torch.diag(torch.tensor(np.concatenate(
#                                     [np.repeat((self.scale_force**2) *
#                                                self.dt**2,2), 
#                                      np.repeat(self.scale_aux**2,4)]
#                                     )))
#         # ToDo: in DTC_2D I define these as nn.Parameters.
#         # didn't do it here because maybe these should be defined globally.
#         # could appear also in emission and potentially proposal.
#         self.L1 = inits_dict["L1"] # upper arm length
#         self.L2 = inits_dict["L2"] # forearm length
#         self.M1 = inits_dict["M1"] # upper arm mass
#         self.M2 = inits_dict["M2"] # forearm mass
#         self.g = g # gravity constant, use only later
#         self.dim_latents = 6

#     def D(self, t2):
#         '''computes inertia tensor from static parameters and the current angle t2.
#         in a deterministic world without batches and particles, this is a 2X2 matrix.
#         Args: 
#             t2: current elbow angle. torch.Tensor(batch_size * num_particles)
#         Returns:
#             torch.tensor [batch_size*num_particles, 2, 2]. These dimensions are
#             important [large_batch_size, [squared mat]] 
#             because they allow us to later invert D'''
#         D_tensor = torch.zeros(t2.shape[0], 2, 2, dtype = torch.double)
#         D_tensor[:,0,0] = self.L1**2*self.M1/3 + self.M2*(3*self.L1**2 + 
#                             3*self.L1*self.L2*torch.cos(t2) + self.L2**2)/3
#         D_tensor[:,0,1] = self.L2*self.M2*(3*self.L1*torch.cos(t2) + 2*self.L2)/6
#         D_tensor[:,1,0] = self.L2*self.M2*(3*self.L1*torch.cos(t2) + 2*self.L2)/6
#         D_tensor[:,1,1] = self.L2**2*self.M2/3*torch.ones_like(t2)
        
#         return D_tensor
    
#     @staticmethod
#     def Newton_2nd(torque_vec_tens, 
#                    D_mat_tens, h_vec_tens, c_vec_tens):
#         # ToDo: think whether we want to log accel.
#         '''compute instantaneous angular acceleration, a length-2 column vector, 
#         according to Newton's second law: 
#             force = mass*acceleration, 
#         which in the angular setting (and in 2D), becomes: 
#             torque (2X1) = intertia_tens (2X2) * angular_accel (2X1)
#         Here, we are multiplying both sides of the equation by 
#         intertia_tens**(-1) to remain with an expression for angular acceleration.
#         We are also taking into account "fictitious forces" coriolis/centripetal and gravity
#         forces. 
#         The function executes three operations:
#             (1) invert the intertia tensor, 
#             (2) subtract "fictitious forces" from torque vector
#             (3) compute the dot product between (1) and (2)
        
#         Args: 
#             torque_vec_tens: latent forces from the last time point, 
#                 torch.tensor [batch_size*num_particles, 2, 1].
#                 ToDo: make sure it is viewed that way.
            
#             D_mat_tens: output of self.D, inertia tensor that is computed from
#                 static parameters and angles.
#                 torch.tensor [batch_size*num_particles, 2, 2]. These dimensions are
#                 important [batch_size * num_particles, [squared mat]] 
#                 because they allow us to later invert D
            
#             h_vec_tens: output of self.h_vec, Coriolis and centripetal vector, 
#                 computed from static params and 
#                 instantaneus angular velocities and elbow angle.
#                 torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
#                 dot product later; should be a length-2 column vector.
            
#             c_vec_tens: output of self.c_vec, gravity vector.
#                 computed from static parameters, current configuration of both angles 
#                 and gravity constant.
#                 torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
#                 dot product later; should be a length-2 column vector.
            
#         Returns:
#             torch.Tensor([batch_size * num_particles, 2]). Should be a length-2 column vector
#             if we have two angles -- matching the angular velocity and angle vectors.
#             TODO: make sure that this tensor is logged when this function is called.
#             potentially use wrapper that logs it inside emission. 
#             in general, consider logging h_vec and c_vec
#             '''
#         D_inv_mat_tens = torch.inverse(D_mat_tens)
#         brackets = torque_vec_tens - h_vec_tens - c_vec_tens 
#         inst_accel = D_inv_mat_tens.matmul(brackets)
#         # want the output of this to be 
#         # size [batch_size, num_part, 2]. we view it differently outside
#         return inst_accel # inst_accel.squeeze()
    
#     def forward(self, previous_latents=None, time=None,
#                 previous_observations=None):
        
#         batch_size = previous_latents[-1].shape[0]
#         num_particles = previous_latents[-1].shape[1]
#         diag_expanded = self.diag_mat.expand(
#                     batch_size,num_particles,
#                     self.dim_latents,
#                     self.dim_latents)
#         # compute a(x_{k-1}) of shape (batch_size*num_particles,6)
#         ax = torch.zeros([batch_size, 
#                           num_particles, 
#                           self.dim_latents], dtype = torch.double) # save room

#         ax[:,:,2] = previous_latents[-1][:,:,4] # \dot{theta}_1
#         ax[:,:,3] = previous_latents[-1][:,:,5] # \dot{theta}_2
        
#         # fill in the last two entries with Newton's second law

#         inert_tens = self.D(previous_latents[-1][:,:,3].\
#                                   view(batch_size*num_particles)) # 4th element in state
#         torque_vec = previous_latents[-1][:,:,:2].\
#             view(batch_size*num_particles,2,1) # first two elements in state vec
#         # for now, no gravity and fictitious forces
#         c_vec = torch.zeros_like(torque_vec)
#         h_vec = torch.zeros_like(torque_vec)
        
#         # compute accel using Netwon's second law
#         accel = self.Newton_2nd(torque_vec, 
#                                       inert_tens, h_vec, c_vec)
        
#         ax[:,:,4:] = accel.view(batch_size, num_particles, 2)

#         # deterministic first order Euler integration
#         mean_fully_expanded = previous_latents[-1] + self.dt * ax

#         # return distribution
#         return aesmc.state.set_batch_shape_mode(
#             torch.distributions.MultivariateNormal(
#             mean_fully_expanded, diag_expanded),
#             aesmc.state.BatchShapeMode.FULLY_EXPANDED)

# in previous codes: Transition_Short
class Transition(nn.Module): 
    '''x \in R^6 := [tau_1, tau_2, theta_1, theta_2, 
    \dot{theta}_1, \dot{theta}_2]^T'''
    def __init__(self, dt, 
                 scale_force, 
                 scale_aux, 
                 arm_class_instance):
        super(Transition, self).__init__()
        self.arm_model = arm_class_instance
        self.dt = dt
        self.scale_force = scale_force 
        self.scale_aux = scale_aux # scale aux should be smaller than dt
        self.diag_mat = torch.diag(torch.tensor(np.concatenate(
                                    [np.repeat((self.scale_force**2) *
                                               self.dt**2,2), 
                                     np.repeat(self.scale_aux**2,4)]
                                    )))
        self.dim_latents = 6
    
    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        
        batch_size = previous_latents[-1].shape[0]
        num_particles = previous_latents[-1].shape[1]
        diag_expanded = self.diag_mat.expand( # ToDo: check move to _init_
                    batch_size,num_particles,
                    self.dim_latents,
                    self.dim_latents)

        mean_fully_expanded = self.arm_model.forward(previous_latents)

        # return distribution
        return aesmc.state.set_batch_shape_mode(
            torch.distributions.MultivariateNormal(
            mean_fully_expanded, diag_expanded),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)
    
class Transition_Velocity(nn.Module): 
    '''x \in R^4 := [theta_1, theta_2, 
    \dot{theta}_1, \dot{theta}_2]^T'''
    def __init__(self, dt, 
                 scale_accel, 
                 scale_aux):
        super(Transition_Velocity, self).__init__()
        self.dt = dt
        self.scale_accel = scale_accel 
        self.scale_aux = scale_aux # scale aux should be smaller than dt
        self.diag_mat = torch.diag(torch.tensor(np.concatenate(
                                    [np.repeat(self.scale_aux**2
                                               ,2), 
                                     np.repeat((self.scale_accel**2) * 
                                               self.dt**2, 2)]
                                    )))
        self.dim_latents = 4
    
    def forward(self, previous_latents=None, time=None,
                previous_observations=None):
        
        batch_size = previous_latents[-1].shape[0]
        num_particles = previous_latents[-1].shape[1]
        diag_expanded = self.diag_mat.expand( # ToDo: check move to _init_
                    batch_size,num_particles,
                    self.dim_latents,
                    self.dim_latents)
        
        xdot = torch.zeros(batch_size, num_particles, self.dim_latents)
        xdot[:,:,:2] = previous_latents[-1][:,:,2:]
        
        mean_fully_expanded = previous_latents[-1] + self.dt * xdot

        # return distribution
        return aesmc.state.set_batch_shape_mode(
            torch.distributions.MultivariateNormal(
            mean_fully_expanded, diag_expanded),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)

class Emission_Linear(nn.Module):
    """y_t = Cx_{t} + v_t, v_t \sim N(0,R)"""
    def __init__(self, C, R):
        super(Emission_Linear, self).__init__()
        self.C = torch.tensor(C) # emission mat. dim obs X dim latents
        self.R = torch.tensor(R) # cov mat. dim obs X dim obs
        self.dim_latents = self.C.shape[1]
        self.dim_obs = self.C.shape[0]
        assert(self.C.shape[0] == self.R.shape[0])

    def forward(self, latents=None, time=None, previous_observations=None):
        
        batch_size = latents[-1].shape[0]
        num_particles = latents[-1].shape[1]
        R_fully_expanded = self.R.expand(batch_size, num_particles,
                                   self.dim_obs, self.dim_obs)
        C_batch_expanded = self.C.expand(batch_size * num_particles,
                                   self.dim_obs, self.dim_latents)

        # compute Gx_{t}
        mean_fully_expanded = C_batch_expanded.matmul(
                            latents[-1].contiguous().view(
                                -1, self.dim_latents, 1)).view(
                                    batch_size,num_particles,-1)

        # return distribution y_t = Gx_{t} + v_t, v_t \sim N(0,R)
        return aesmc.state.set_batch_shape_mode(
            torch.distributions.MultivariateNormal(
            mean_fully_expanded, R_fully_expanded),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)
    
class Emission(nn.Module):
    '''ToDo: note that all the other objects work with torch.double precision.
    make sure it's the same here.'''
    """This Emission just picks theta_1 and (latents[-1][:,:,2]) theta_2
    (latents[-1][:,:,3])
    (latents[-1][:,:,1]), and adds noise. later will include 2D FW KIN """
    def __init__(self, inits_dict, cov_mat, theta_indices):
        super(Emission, self).__init__()
        self.cov_mat = torch.tensor(cov_mat) # should be nn.Parameter in the future
        self.L1 = inits_dict["L1"]
        self.L2 = inits_dict["L2"]
        self.theta_indices = theta_indices # should be a list, indicating which
        # elements of the state vectore are theta.

    def FW_kin_2D(self, angles):
        ''' Forward kinematics function that transforms angles to coordinates
            at a single time point.
            Args:
                angles: torch.tensor [batch_size, num_particles, n_latent_dim]
                this is typically the current values of the angles.
            Returns:
                coords: torch.tensor [batch_size, num_particles, n_obs_dim]'''
        coords = torch.stack(
            (
            torch.zeros_like(angles[:,:,0]), # x_0
            torch.zeros_like(angles[:,:,0]), # y_0
            self.L1*torch.cos(angles[:,:,0]), # x_1
            self.L1*torch.sin(angles[:,:,0]), # y_1
            self.L2*torch.cos(angles[:,:,0]+angles[:,:,1]) + 
            self.L1*torch.cos(angles[:,:,0]), # x_2
            self.L2*torch.sin(angles[:,:,0]+angles[:,:,1]) + 
            self.L1*torch.sin(angles[:,:,0]), # y_2
            ), 
            dim=2)
        return coords

    def forward(self, latents=None, time=None, previous_observations=None):
        # fw-kin
        
        # pick these two elements, working
        #mean_tensor = self.FW_kin_2D(latents[-1][:,:,[2,3]])
        
        # pick two thetas
        mean_tensor = self.FW_kin_2D(latents[-1][:,:,self.theta_indices])


        return aesmc.state.set_batch_shape_mode( 
            torch.distributions.MultivariateNormal(mean_tensor, self.cov_mat),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)

# in previous models, this was called Bootstrap_Proposal_Short
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
    '''this is an optimal proposal for a non-linear transition linear
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

# class Bootstrap_Proposal(nn.Module):
#     """This proposal is proportional to the transition.
#     at step zero, the proposal should be set to the initial distribution
#     at step t, the proposal should be set to the transition
#     Args: ToDo: update
#         scale_0, scale_t: scalars for __init__ method
#         previous_latents: list of len num_timesteps, each entry is 
#             torch.tensor([batch_size, num_particles, dim_latents])
#         time: integer
#     Returns:
#         torch.distributions.Normal object. 
#         at time=0, torch.tensor([batch_size, dim_latents]) 
#         and at time!=0, torch.tensor([batch_size, num_particles, dim_latents])"""
#     def __init__(self, dt, inits_dict, mu_0, cov_0, 
#                  scale_force, scale_aux, g):
#         super(Bootstrap_Proposal, self).__init__()
#         self.dt = dt
#         self.mu_0 = torch.tensor(mu_0, dtype = torch.double)
#         self.cov_0 = torch.tensor(cov_0, dtype = torch.double)
#         self.scale_force = scale_force # currently not used, i.e. assume = 1
#         self.scale_aux = scale_aux # scale aux should be smaller than dt, or add it to every entry
#         self.diag_mat = torch.diag(torch.tensor(np.concatenate(
#                                     [np.repeat((self.scale_force**2) *
#                                                self.dt**2,2), 
#                                      np.repeat(self.scale_aux**2,4)]
#                                     )))
#         # ToDo: in DTC_2D I define these as nn.Parameters.
#         # didn't do it here because maybe these should be defined globally.
#         # could appear also in emission and potentially proposal.
#         self.L1 = inits_dict["L1"] # upper arm length
#         self.L2 = inits_dict["L2"] # forearm length
#         self.M1 = inits_dict["M1"] # upper arm mass
#         self.M2 = inits_dict["M2"] # forearm mass
#         self.g = g # gravity constant, use only later
#         self.dim_latents = 6
    
#     def D(self, t2):
#         '''computes inertia tensor from static parameters and the current angle t2.
#         in a deterministic world without batches and particles, this is a 2X2 matrix.
#         Args: 
#             t2: current elbow angle. torch.Tensor(batch_size * num_particles)
#         Returns:
#             torch.tensor [batch_size*num_particles, 2, 2]. These dimensions are
#             important [large_batch_size, [squared mat]] 
#             because they allow us to later invert D'''
#         D_tensor = torch.zeros(t2.shape[0], 2, 2, dtype = torch.double)
#         D_tensor[:,0,0] = self.L1**2*self.M1/3 + self.M2*(3*self.L1**2 + 
#                             3*self.L1*self.L2*torch.cos(t2) + self.L2**2)/3
#         D_tensor[:,0,1] = self.L2*self.M2*(3*self.L1*torch.cos(t2) + 2*self.L2)/6
#         D_tensor[:,1,0] = self.L2*self.M2*(3*self.L1*torch.cos(t2) + 2*self.L2)/6
#         D_tensor[:,1,1] = self.L2**2*self.M2/3*torch.ones_like(t2)
        
#         return D_tensor
    
#     @staticmethod
#     def Newton_2nd(torque_vec_tens, 
#                    D_mat_tens, h_vec_tens, c_vec_tens):
#         # ToDo: think whether we want to log accel.
#         '''compute instantaneous angular acceleration, a length-2 column vector, 
#         according to Newton's second law: 
#             force = mass*acceleration, 
#         which in the angular setting (and in 2D), becomes: 
#             torque (2X1) = intertia_tens (2X2) * angular_accel (2X1)
#         Here, we are multiplying both sides of the equation by 
#         intertia_tens**(-1) to remain with an expression for angular acceleration.
#         We are also taking into account "fictitious forces" coriolis/centripetal and gravity
#         forces. 
#         The function executes three operations:
#             (1) invert the intertia tensor, 
#             (2) subtract "fictitious forces" from torque vector
#             (3) compute the dot product between (1) and (2)
        
#         Args: 
#             torque_vec_tens: latent forces from the last time point, 
#                 torch.tensor [batch_size*num_particles, 2, 1].
#                 ToDo: make sure it is viewed that way.
            
#             D_mat_tens: output of self.D, inertia tensor that is computed from
#                 static parameters and angles.
#                 torch.tensor [batch_size*num_particles, 2, 2]. These dimensions are
#                 important [batch_size * num_particles, [squared mat]] 
#                 because they allow us to later invert D
            
#             h_vec_tens: output of self.h_vec, Coriolis and centripetal vector, 
#                 computed from static params and 
#                 instantaneus angular velocities and elbow angle.
#                 torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
#                 dot product later; should be a length-2 column vector.
            
#             c_vec_tens: output of self.c_vec, gravity vector.
#                 computed from static parameters, current configuration of both angles 
#                 and gravity constant.
#                 torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
#                 dot product later; should be a length-2 column vector.
            
#         Returns:
#             torch.Tensor([batch_size * num_particles, 2]). Should be a length-2 column vector
#             if we have two angles -- matching the angular velocity and angle vectors.
#             TODO: make sure that this tensor is logged when this function is called.
#             potentially use wrapper that logs it inside emission. 
#             in general, consider logging h_vec and c_vec
#             '''
#         D_inv_mat_tens = torch.inverse(D_mat_tens)
#         brackets = torque_vec_tens - h_vec_tens - c_vec_tens 
#         inst_accel = D_inv_mat_tens.matmul(brackets)
#         # want the output of this to be 
#         # size [batch_size, num_part, 2]. we view it differently outside
#         return inst_accel # inst_accel.squeeze()
    
#     def forward(self, previous_latents=None, time=None, observations=None):
        
#         if time == 0: # initial
#             return aesmc.state.set_batch_shape_mode(
#                 torch.distributions.MultivariateNormal(
#                     self.mu_0, self.cov_0), 
#                 aesmc.state.BatchShapeMode.NOT_EXPANDED)
 
#         else: # transition
            
#             batch_size = previous_latents[-1].shape[0]
#             num_particles = previous_latents[-1].shape[1]
#             diag_expanded = self.diag_mat.expand(
#                 batch_size,num_particles, 
#                 self.dim_latents, 
#                 self.dim_latents)
            
#             # start: compute a(x_{k-1}) of shape (batch_size*num_particles,6)
#             ax = torch.zeros([batch_size, 
#                           num_particles, 
#                           self.dim_latents], dtype = torch.double) # save room

    
#             ax[:,:,2] = previous_latents[-1][:,:,4] # \dot{theta}_1
#             ax[:,:,3] = previous_latents[-1][:,:,5] # \dot{theta}_2
            
#             # fill in the last two entries with Newton's second law
#             inert_tens = self.D(previous_latents[-1][:,:,3].\
#                                       view(batch_size*num_particles)) # 4th element in state
#             torque_vec = previous_latents[-1][:,:,:2].\
#                 view(batch_size*num_particles,2,1) # first two elements in state vec
#             # for now, no gravity and fictitious forces
#             c_vec = torch.zeros_like(torque_vec)
#             h_vec = torch.zeros_like(torque_vec)
            
#             # compute accel using Netwon's second law
#             accel = self.Newton_2nd(torque_vec, 
#                                           inert_tens, h_vec, c_vec)
            
#             ax[:,:,4:] = accel.view(batch_size, num_particles, 2)
#             # end: compute a(x_{k-1})
#             # deterministic first order Euler integration
#             mean_fully_expanded = previous_latents[-1] + self.dt * ax
    
#             # return distribution
#             return aesmc.state.set_batch_shape_mode(
#                 torch.distributions.MultivariateNormal(
#                 mean_fully_expanded, diag_expanded), # float() was important
#                 aesmc.state.BatchShapeMode.FULLY_EXPANDED)


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
        self.lin_0 = nn.Sequential(
                        nn.Linear(6, 2),
                        nn.ReLU()) # observations[0] -> mu[0]
        self.lin_t = nn.Sequential(
                        nn.Linear(14, 2), # SHOULD BE 6,2 IN EULER
                        nn.ReLU()) # {previous_latents[t-1], observations[t-1], observations[t]} -> mu[t]

    def forward(self, previous_latents=None, time=None, observations=None):
        if time == 0:
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.lin_0(observations[0]),
                    scale=self.scale_0),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        else:
            if time == 1:
                self.batch_size, self.num_particles, self.dim_latents = previous_latents[-1].shape
            expanded_obs_prev = aesmc.state.expand_observation(observations[time-1], self.num_particles) # expand current obs
            expanded_obs = aesmc.state.expand_observation(observations[time], self.num_particles) # expand current obs
            # concat previous latents with current expanded observation. then squeeze to batch_expanded mode
            # i.e., [batch_size*num_particles, dim_latent+dim_observation]
            # to apply a linear layer.
            concat_squeezed = torch.cat([previous_latents[-1],
                                         expanded_obs_prev,
                        expanded_obs
                        ], 
                        dim=2).view(-1, previous_latents[-1].shape[2]+
                                    expanded_obs.shape[2] +
                                    expanded_obs_prev.shape[2] )
            activ = self.lin_t(concat_squeezed)
            mu_t = activ.view(self.batch_size, self.num_particles, self.dim_latents)
            
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=mu_t,
                    scale=self.scale_t),
                aesmc.state.BatchShapeMode.FULLY_EXPANDED)        

class TrainingStats(object):
    def __init__(self, M1_true, M2_true,
                 num_timesteps,
                 logging_interval=100):
        self.M1_true = M1_true
        self.M2_true = M2_true
        self.logging_interval = logging_interval
        self.curr_M1 = []
        self.curr_M2 = []
        self.loss = []
  
    def __call__(self, epoch_idx, epoch_iteration_idx, loss, initial,
                 transition, emission, proposal):
        if epoch_iteration_idx % self.logging_interval == 0:
            print('Iteration {}: Loss = {}'.format(epoch_iteration_idx, loss))
            self.curr_M1.append(transition.arm_model.M1.item())
            self.curr_M2.append(transition.arm_model.M2.item()) # seemed to be better than detach().numpy()
            self.loss.append(loss)
            # print(np.round(np.linalg.norm(
            #     np.array([emission.L1.detach().numpy(),emission.L2.detach().numpy()])-
            #     np.array([self.L1_true, self.L2_true])),2))
            
            
            
test = False
if test:
    # ToDo: keep exploring the dimensions of things and how my reshape effects
    # the forward kin function, etc.
    
    
    batch_size = 10
    num_particles = 100
    dt = 0.03
    scale=1
    
    initial = Initial(0.7, 0.5)
    initial().sample()
    diag_mat = torch.diag(torch.tensor([0.03, 0.01, 0.01]))
    diag_expanded = diag_mat.expand(batch_size,num_particles,3,3)
    fake_prev_latents = [torch.tensor([3.0, 7.0, 1.2]).expand(batch_size, num_particles, 3)]
    fake_prev_latents[-1].shape
    multi_trans_dist = torch.distributions.MultivariateNormal(
        fake_prev_latents[-1], diag_expanded)
    print(multi_trans_dist)
    val = multi_trans_dist.sample()
    val.shape
    print(val[2,4,:])
    lp = multi_trans_dist.log_prob(val)
    print(lp[:2,:3])
    
    # check transition
    batch_size = 10
    num_particles = 100
    dt = 0.03
    inits_dict = {}
    inits_dict["L1"] = 1.0
    inits_dict["L2"] = 1.0
    inits_dict["M1"] = 0.5
    inits_dict["M2"] = 0.5
    scale_force = 1
    scale_aux = 0.01
    g = 0.01
    # define instance
    transition = Transition(0.01, inits_dict, scale_force, scale_aux, g)

    # mimicking working with a vector of previous latents.
    fake_prev_latents = [torch.tensor([10.0, 1.0, 1.2, 2.1, 0.1, 0.5] \
                                      ).expand(batch_size, num_particles, 6)]
    print(fake_prev_latents[-1].shape)
    t2 = fake_prev_latents[-1][:,:,3].view(batch_size*num_particles) # ind 3 means x_4
    print(t2.shape)
    inert_tens = transition.D(t2)
    print(inert_tens.shape)
    # now , check netwon's second
    torque_vec = fake_prev_latents[-1][:,:,:2].\
        view(batch_size*num_particles,2,1)
    print(torque_vec.shape)
    c_vec = torch.zeros_like(torque_vec).float()
    h_vec = torch.zeros_like(torque_vec)
    accel = transition.Newton_2nd(torque_vec, 
                                  inert_tens, h_vec, c_vec)
    print(accel.shape)
    # we should resize the output of this
    print(accel.view(batch_size, num_particles, 2).shape) 
    
    ax = torch.zeros_like(fake_prev_latents[-1])
    ax[:,:,4:] = accel.view(batch_size, num_particles, 2)
    print(ax[2,13,:])
    print(torque_vec.view(batch_size, num_particles, 2)[2,13,:])
    
    one_trans_call  = transition(fake_prev_latents)
    one_trans_call.sample().shape
    
    # now let's define emission
    emission = Emission(0.03)
    one_call = emission(fake_prev_latents)
    one_call
    one_call.sample().shape
    
    # check proposal
    proposal = Bootstrap_Proposal(dt, inits_dict, 1.0, 
                                  0.0, scale_force, scale_aux, g)
    prop_0 = proposal(time=0, observations = [torch.ones(batch_size,2)])
    prop_0.sample().shape
    prop_1 = proposal(time=1, previous_latents = fake_prev_latents, 
                      observations = [torch.ones(batch_size,
                                                         num_particles, 2)])
    prop_1.sample().shape
    prop_1
    ax = torch.zeros([batch_size, num_particles, 3])
    ax[:,:,1] = fake_prev_latents[-1][:,:,2]
    ax[:,:,2] = fake_prev_latents[-1][:,:,0] * (1.0/2)
    print(ax[3, 0, :]) # should be 0, 1.2, 1.5
    # deterministic first order Euler integration
    mean_fully_expanded = fake_prev_latents[-1] + dt * ax
    #fake_prev_latents = [torch.zeros(batch_size, num_particles,2)] # list element 1
    fake_prev_latents[-1].view(-1,2).shape
    A = torch.tensor([[1, dt],[0,1]], requires_grad = False)
    
    Q = torch.tensor([[0.25*(dt**4), 0.5*(dt**3)], 
                               [ 0.5*(dt**3), dt**2]], 
                              requires_grad = False)*scale # this is 2X2
    Q.shape
    mean_batch_expanded = A.expand(batch_size*num_particles, 2,2).matmul(
                    fake_prev_latents[-1].view(-1,2,1)).view(batch_size*num_particles,-1)
    mean_fully_expanded = A.expand(batch_size*num_particles, 2,2).matmul(
                    fake_prev_latents[-1].view(-1,2,1)).view(batch_size,num_particles,-1)
   
    scale_batch_expanded = Q.expand(batch_size*num_particles, 2,2)
    torch.ones(batch_size*num_particles,2,2)
    scale_batch_expanded.shape
    little_scale = torch.tensor([1,1])
    dist = torch.distributions.MultivariateNormal(
                torch.ones(2), scale_tril = torch.diag(little_scale))
    
    G = torch.tensor([[0.5*(dt**2)], [dt]], 
                              requires_grad = False) * scale # this is for 2d cast
    dist = torch.distributions.lowrank_multivariate_normal.\
    LowRankMultivariateNormal(mean_fully_expanded, #torch.tensor([4,7], dtype = torch.float), 
                              G, torch.zeros(2)) # seems to work
    
    dist2 =  torch.distributions.lowrank_multivariate_normal.\
    LowRankMultivariateNormal(torch.tensor([4,7], dtype = torch.float), 
                              torch.tensor([1,0], dtype = torch.float).view(2,1), 
        torch.tensor([0.01, 0.01]))
    val = dist2.sample()
    print(val)
    lp = dist2.log_prob(val)
    print(lp) # because it's a multinormal, we have one scalar for log prob.
    # here's also a degenerate distribution that might work in our case
    # note that the numbers are floats, that is important.
    # first entry in cov could be either 0.0 (we'll get a nan logprob) or 0.01
    # however we get multiple, i.e., two log probs in this case
    dist3 = torch.distributions.Normal(torch.tensor([1.3, 2.7]), 
                                       torch.tensor([0.01, 1.1]))
    val = dist3.sample()
    print(val)
    lp = dist3.log_prob(val)
    print(lp)
    
    # check transiotion
    transition = Transition(dt, scale)
    dist = transition.forward(fake_prev_latents, time = 1)
    val = dist.sample()
    print(val[0,:,:])
    
    # check something inside emission
    mean = fake_prev_latents[-1][:,:,0].view(
            fake_prev_latents[-1].shape[0], fake_prev_latents[-1].shape[1], 1)
    
    # check emission
    emission = Emission(0.01)
    emission_dist = emission([val])
    emit_val = emission_dist.sample()
    print(emit_val[1,34,0])
    print(val[1,34,0]) # these two should be close
    
    m = torch.distributions.lowrank_multivariate_normal.\
        LowRankMultivariateNormal(torch.tensor([1,2]), 
                                  G, 
                                  torch.zeros(2))
    inits_dict = {}
    inits_dict["L1"] = 1.0
    inits_dict["L2"] = 1.0
    inits_dict["M1"] = 0.5
    inits_dict["M2"] = 0.5
    # note, if doesn't work, could add 1 in last dim
    inits_dict["velocity_vec"] = torch.zeros([batch_size*num_particles, 2]) # could have 1 in last dim
    inits_dict["angle_vec"] = torch.rand([batch_size*num_particles, 2])
    
    # ToDo: onsider having consts dict
    g= 0.2
    dt=0.03
    
    emission = Emission(inits_dict, dt, g, 0.01, False)
    print(list(emission.parameters()))
    big_bs = 30
    tens_ones = torch.ones(big_bs)
    D_tens_test = emission.D(tens_ones) # keep for testing Newton
    #print(D_tens_test)
    print(D_tens_test.shape) 
    
    # torch.t(torch.ones(3,1)).mm(torch.ones(3,1)*2)
    # a = torch.tensor([[1.0],[2.0]]).expand()
    # inv = torch.inverse(D_tens_test)
    # res = inv.mm(a) 
    
    # check h_vec
    
    # one timestep
    res = emission.h_vec(torch.tensor([0.6]), torch.tensor([0.2]), torch.tensor([0.3]))
    res.shape
    
    # multiple timesteps
    a = torch.ones(30)
    b = torch.ones_like(a)*2
    #b = torch.ones(31) # should fail assertion
    c = torch.ones_like(a)*3
    h_vec = emission.h_vec(a, b, c) # keep for testing Newton
    h_vec.shape
    
    # check c_vec
    a = torch.ones(30)
    b = torch.ones_like(a)*2
    #b = torch.ones(31) # should fail assertion
    c_vec = emission.c_vec(a,b) # keep for testing Newton
    c_vec.shape
    
    latent_torque = torch.ones([30,2,1])
    latent_torque.shape
    
    # check transition velocity dynamics.
    batch_size = 10
    num_particles = 100
    dt = 0.03
    inits_dict = {}
    inits_dict["L1"] = 1.0
    inits_dict["L2"] = 1.0
    inits_dict["M1"] = 0.5
    inits_dict["M2"] = 0.5
    scale_accel = 10.0
    scale_aux = 0.01
    
    # define instance
    transition = Transition_Velocity(dt, scale_accel, scale_aux)

    # mimicking working with a vector of previous latents.
    fake_prev_latents = [torch.tensor([10.0, 1.0, 1.2, 2.1] \
                                      ).expand(batch_size, num_particles, 4)]
    
    transition(fake_prev_latents).covariance_matrix[0,0,:,:].detach().numpy()
    
    #how to extract angles for c and h
    t1 = fake_prev_latents[-1][:,:,2].\
                              contiguous().view(batch_size*num_particles)
    t2 = fake_prev_latents[-1][:,:,3].\
                              contiguous().view(batch_size*num_particles)
    dt1 = fake_prev_latents[-1][:,:,4].\
                              contiguous().view(batch_size*num_particles)
    dt2 = fake_prev_latents[-1][:,:,5].\
                              contiguous().view(batch_size*num_particles)
    
    c_vec = torch.zeros_like(torque_vec)
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:40:13 2020

@author: danbiderman
"""


import copy
import aesmc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Initial: # distribution latent transition at t=0
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
    '''Emission function that.
        '''
    def __init__(self, inits_dict, dt, g, scale, learn_static):
        super(Emission, self).__init__()
        '''Args:
            inits_dict: dictionary with float entries for L1, L2, M1, M2.
            scale: float, currently fixed quantity that defines scale of normal dist
            dt: float, fixed, for Euler steps.
            learn_static: True/False, whether or not to learn the static parameters
                initialized by inits_dict'''
        # nn.Parameters are useful because they are tensors that automatically 
        # appear in Emission.parameters(). see https://stackoverflow.com/questions/50935345/understanding-torch-nn-parameter 
        # without the squeeze() each one would be torch.Size([1]). 
        # ToDo: see what happens without the squeeze in the larger script.
        
        self.L1 = nn.Parameter(torch.Tensor([inits_dict["L1"]]).squeeze(), 
                               requires_grad = learn_static) # upper arm length
        self.L2 = nn.Parameter(torch.Tensor([inits_dict["L2"]]).squeeze(), 
                               requires_grad = learn_static) # forearm length
        self.M1 = nn.Parameter(torch.Tensor([inits_dict["M1"]]).squeeze(), 
                               requires_grad = learn_static) # upper arm mass
        self.M2 = nn.Parameter(torch.Tensor([inits_dict["M2"]]).squeeze(), 
                       requires_grad = learn_static) # forearm mass
        # ToDo: we currently have no flexibility with values
        # I thought about including them in the forward method but then
        # the internal inference and train modules would have to be adapted,
        # which I want to avoid.
        self.init_velocity = inits_dict["velocity_vec"] 
        self.init_angles = inits_dict["angle_vec"]
        self.scale = scale # ToDo: should be nn.Parameter in the future
        self.dt = dt 
        self.g = g
        
        # add batch size and num particles?

        
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
    
    
    def D(self, t2):
        '''computes inertia tensor from static parameters and the current angle t2.
        in a deterministic world without batches and particles, this is a 2X2 matrix.
        Args: 
            t2: current elbow angle. torch.Tensor(batch_size * num_particles)
        Returns:
            torch.tensor [batch_size*num_particles, 2, 2]. These dimensions are
            important [large_batch_size, [squared mat]] 
            because they allow us to later invert D'''
        D_tensor = torch.zeros(t2.shape[0], 2, 2)
        D_tensor[:,0,0] = self.L1**2*self.M1/3 + self.M2*(3*self.L1**2 + 
                            3*self.L1*self.L2*torch.cos(t2) + self.L2**2)/3
        D_tensor[:,0,1] = self.L2*self.M2*(3*self.L1*torch.cos(t2) + 2*self.L2)/6
        D_tensor[:,1,0] = self.L2*self.M2*(3*self.L1*torch.cos(t2) + 2*self.L2)/6
        D_tensor[:,1,1] = self.L2**2*self.M2/3*torch.ones_like(t2)
        
        return D_tensor
    
    def h_vec(self, dt1, dt2, t2):
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
    
    def c_vec(self, t1, t2):
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
    
    def Newton_2nd(self, torque_vec_tens, 
                   D_mat_tens, h_vec_tens, c_vec_tens):
        # ToDo: could also be a static method?
        # ToDo2: think how we're logging quantities
        # here.
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
            ToDo: make sore that this tensor is logged when this function is called.
            potentially use wrapper that logs it inside emission. 
            '''
        D_inv_mat_tens = torch.inverse(D_mat_tens)
        brackets = torque_vec_tens - h_vec_tens - c_vec_tens # check
        inst_accel = D_inv_mat_tens.matmul(brackets)
        
        return inst_accel.squeeze()
    
    #ToDo: static method as well?
    #ToDo2: wrap and log in the same manner
    def Euler_step(self, variable, variable_derivative):
        '''given the value of a variable and its derivative at time t, 
        take an Euler step (first order linear approx) to predict value at time t+dt.
        Args:
            variable: torch.tensor([batch_size*num_particles, dim_var])
            variable_derivative: torch.tensor([batch_size*num_particles, dim_var])
        Returns:
            torch.tensor([batch_size*num_particles, dim_var, 1])
        '''
        return variable + self.dt * variable_derivative
    
    def forward(self, latents = None, time = None, previous_observations = None):
        
        # time 0
        # use initial angles to explain data
        # one could optimize these angles and velocities to start from a good initial point
        #print("latent: " , latents[-1].shape)
        if time == 0:
            
            print('itreration 0: length of latent[-1] = ', len(latents))
            print('itreration 0: shape of latents[-1] = ', latents[-1].shape)

            # compute dimensions 
            self.batch_size = latents[-1].shape[0]
            self.num_particles = latents[-1].shape[1]
            self.dim_latents = latents[-1].shape[2]
            
            
            # set lists at the first time step and then append values
            self.angles = [] # append inits
            self.velocity = [] # append inits.
            self.acceleration = [] # append just in the next time step
            
            init_angles = torch.zeros_like(latents[-1]).view(-1,2)
            init_velocity = torch.zeros_like(latents[-1]).view(-1,2)
            # compute acccel, rhs for current time. use inits and sampled torque.
            accel_curr = self.Newton_2nd(
                latents[-1].contiguous().view( # squeezing. ToDo: not sure about line.
                                        self.batch_size*self.num_particles, 2, 1), # self.batch_size*self.num_particles ,2, 1),
                                        self.D(init_angles[:,1]), 
                                        self.h_vec(init_velocity[:,0], 
                                                   init_velocity[:,1], 
                                                   init_angles[:,1]), 
                                        self.c_vec(init_angles[:,0], 
                                                   init_angles[:,1]))
            
            self.acceleration.append(accel_curr) # append first value

            # compute current velocity using current accel (just appended) and init angle.
            velocity_curr = self.Euler_step(init_velocity, self.acceleration[-1])
            self.velocity.append(velocity_curr)

            # compute current angles using current velocity (just appended) and init angle.
            angles_curr = self.Euler_step(init_angles, self.velocity[-1])
            self.angles.append(angles_curr)
            

        else: # time not zero
            # accel_prev = Newton_2nd(torques, computed_D, computed_c_vec, computed_h_vec)
            # ToDo: check if they have in aesmc a better way to squeeze
            # compute previous acceleration based on previous torques, angles and velocity.
            # note: using latents[-2] because I assume that since then, 
            # we drew the current set of latents in the proposal.
            print('itreration i: length of latent[-1] = ', len(latents))
            print('itreration i: shape of latents[-1] = ', latents[-1].shape)
            print('regular latents: ', latents[-2].shape)
            print('contig latents: ',latents[-2].contiguous().shape)
            print((latents[-2] == latents[-2].contiguous()).detach().numpy().all())
            # ToDo: the part below needs extra caution:
            accel_curr = self.Newton_2nd(
                latents[-1].contiguous().view( # squeezing. ToDo: not sure about line.
                                        self.batch_size*self.num_particles, 2, 1), # self.batch_size*self.num_particles ,2, 1),
                                        self.D(self.angles[-1][:,1]), 
                                        self.h_vec(self.velocity[-1][:,0], 
                                                   self.velocity[-1][:,1], 
                                                   self.angles[-1][:,1]), 
                                        self.c_vec(self.angles[-1][:,0], 
                                                   self.angles[-1][:,1]))
            # append curr accel
            self.acceleration.append(accel_curr)
            # compute current velocity based on previous velocity and current accel.
            velocity_curr = self.Euler_step(self.velocity[-1], self.acceleration[-1])
            self.velocity.append(velocity_curr)

            # compute current angles based on previous angles and velocity.
            angles_curr = self.Euler_step(self.angles[-1], self.velocity[-1])
            self.angles.append(angles_curr)

          
        # for every timestep we conclude with forward kinematics given current angles
        #print("angles: " , self.angles[-1].shape)
            # careful of contiguous!
        mean_tensor = self.FW_kin_2D(self.angles[-1].contiguous().view(
                 self.batch_size, self.num_particles, self.dim_latents))
        
        #print("mean of dist: " , mean_tensor.shape)
            
        return aesmc.state.set_batch_shape_mode(
        torch.distributions.Normal(mean_tensor, self.scale),
        aesmc.state.BatchShapeMode.FULLY_EXPANDED)
            
class Bootstrap_Proposal(nn.Module):
    """This proposal is proportional to the transition.
    at step zero, the proposal should be set to the initial distribution
    at step t, the proposal should be set to the transition
    Args:
        scale_0, scale_t: scalars for __init__ method
        previous_latents: list of len num_timesteps, each entry is 
            torch.tensor([batch_size, num_particles, dim_latents])
        time: integer
    Returns:
        torch.distributions.Normal object. 
        at time=0, torch.tensor([batch_size, dim_latents]) 
        and at time!=0, torch.tensor([batch_size, num_particles, dim_latents])"""
    def __init__(self, scale_0, mu_0, scale_t, mult_t):
        super(Bootstrap_Proposal, self).__init__()
        self.scale_0 = scale_0
        self.mu_0 = mu_0
        self.scale_t = scale_t
        self.mult_t = mult_t

    def forward(self, previous_latents=None, time=None, observations=None):
        if time == 0:
            return aesmc.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.mu_0.expand(observations[-1].shape[0],2), # for a 2d lat
                    scale=self.scale_0),
                aesmc.state.BatchShapeMode.BATCH_EXPANDED)
        else:    
            return aesmc.state.set_batch_shape_mode(
            torch.distributions.Normal(
                self.mult_t * previous_latents[-1], self.scale_t),
            aesmc.state.BatchShapeMode.FULLY_EXPANDED)           
    
# TESTS:===================

test = False
if test:
    
    batch_size = 10
    num_particles = 100
    
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
    
    inst_accel = emission.Newton_2nd(latent_torque, D_tens_test, h_vec, c_vec)
    inst_accel.shape
    inst_accel.squeeze().shape
    (inst_accel.squeeze() == inst_accel[:,:,0]).detach().numpy().all()
    
    velocity = torch.ones([30,2,1])
    next_vel = emission.Euler_step(velocity, inst_accel)
    print(((next_vel - dt*inst_accel)==1).detach().numpy().all()) # should be ones
    
    #test inits
    emission.init_velocity.shape
    emission.init_velocity.view(1000,2).shape

    emission.init_velocity.view(batch_size, num_particles,2).shape
    
    # test forward sweep at t=0
    fake_prev_latents = [torch.zeros(batch_size, num_particles,2)] # element 1
    fake_prev_latents[-1][3,21,0] = 1 # change just a single value from zero
    fake_prev_latents[-1][3,15:30,0]
    # see if the difference is obseved just there
    fake_prev_latents[-1].view(batch_size*num_particles,2,1)
    len(fake_prev_latents)
    fake_prev_latents[-1].shape
    # test forward sweep at t=0
    emission.forward(fake_prev_latents, 0) 
    np.diff(emission.forward(fake_prev_latents, 0).detach().numpy())
    diffs = np.diff(emission.angles[-1].detach().numpy(),axis=0)
    (diffs==0).all() # look at row 321. it is different from prev and next row
    # seems to make sense
    regular_reshape = fake_prev_latents[-1].view(batch_size*num_particles,2).detach().numpy()
    # just 321 should be different
    emission.velocity[-1]
    emission.acceleration[-1] # ToDo: test out case without h_vec/c_vec.
    # test time = 1, first take another fake sample from latent
    fake_prev_latents.append(torch.zeros(batch_size, num_particles,2)) # element 2
    len(fake_prev_latents)
    emission.forward(fake_prev_latents, 1) 
    emission.angles[-1].shape
    len(emission.acceleration)
    emission.acceleration[-1]
    emission.velocity[-1]
    emission.angles[-1]
    emission.acceleration[-1].shape
    fake_prev_latents.append(torch.zeros(batch_size, num_particles,2)) # element 2
    emission.forward(fake_prev_latents, 2) 
    len(emission.acceleration)
    len(emission.velocity)
    emission.forward()
    len(emission.angles)
    emission.velocity[-1][:,:,1].shape
    emission.FW_kin_2D(angles)
    
    fake_prev_latents[-1].shape
    fake_prev_latents.squeeze(1)
    
    fake_obs = [np.zeros()]
    
    transition = Transition(1, 0.2)
    samp = aesmc.state.sample(transition(
                previous_latents=fake_prev_latents, time=2,
                previous_observations=None), batch_size, 1)
    samp.shape
    latents = aesmc.state.sample(transition, batch_size, num_particles)

    def squeeze_num_particles(value):
        if isinstance(value, dict):
            return {k: squeeze_num_particles(v) for k, v in value.items()}
        else:
            return value.squeeze(1)
        
    tuple(map(lambda values: list(map(squeeze_num_particles, values)),
                 [latents, observations]))
# OLDER STUFF BELOW =======================

# def FW_kin_2D(L1, L2, angles):
#     '''Args:
#             L1,L2: scalar quantities or nn.Parameter
#             angles: torch.tensor [batch_size, num_particles, n_latent_dim]
#        Returns:
#             coords: torch.tensor [batch_size, num_particles, n_obs_dim]'''
#     coords = torch.stack(
#         (
#         torch.zeros_like(angles[:,:,0]), # x_0
#         torch.zeros_like(angles[:,:,0]), # y_0
#         L1*torch.cos(angles[:,:,0]), # x_1
#         L1*torch.sin(angles[:,:,0]), # y_1
#         L2*torch.cos(angles[:,:,0]+angles[:,:,1]) + 
#         L1*torch.cos(angles[:,:,0]), # x_2
#         L2*torch.sin(angles[:,:,0]+angles[:,:,1]) + 
#         L1*torch.sin(angles[:,:,0]), # y_2
#         ), 
#         dim=2)
#     return coords

# def D(L1, L2, M1, M2, t2):
#     '''computes inertia tensor from static parameters and the current angle t2.
#     in a deterministic world without batches and particles, this is a 2X2 matrix.
#     Args: 
#         L1: length of upper arm. scalar quantity or torch.nn.Parameter
#         L2: length of fore arm. scalar quantity or torch.nn.Parameter
#         M1: mass of upper arm. scalar quantity or torch.nn.Parameter
#         M2: mass of fore arm. scalar quantity or torch.nn.Parameter
#         t2: current elbow angle. torch.Tensor(batch_size * num_particles)
#     Returns:
#         torch.tensor [batch_size*num_particles, 2, 2]. These dimensions are
#         important [large_batch_size, [squared mat]] 
#         because they allow us to later invert D'''
#     D_tensor = torch.zeros(t2.shape[0], 2, 2)
#     D_tensor[:,0,0] = L1**2*M1/3 + M2*(3*L1**2 + 3*L1*L2*torch.cos(t2) + L2**2)/3
#     D_tensor[:,0,1] = L2*M2*(3*L1*torch.cos(t2) + 2*L2)/6
#     D_tensor[:,1,0] = L2*M2*(3*L1*torch.cos(t2) + 2*L2)/6
#     D_tensor[:,1,1] = L2**2*M2/3*torch.ones_like(t2)
    
#     return D_tensor
#   #   return torch.Tensor([[L1**2*M1/3 + M2*(3*L1**2 + 3*L1*L2*torch.cos(t2) + L2**2)/3, \
#   # L2*M2*(3*L1*torch.cos(t2) + 2*L2)/6], [L2*M2*(3*L1*torch.cos(t2) + 2*L2)/6, L2**2*M2/3]])

# test_args = False
# if test_args:
#     # a way to compress and expand tensors; test case
#     angles = torch.rand([3, 10, 2]) # [batch_size, num_particles, dim_latents]
#     angles2 = angles.view(-1,2)
#     angles3 = angles2.view(-1,10,2)
#     (angles3==angles).detach().numpy().all()
#     # use this manipulation before each time step.
    
#     # test the stacking
#     big_bs = 30
#     tens_ones = torch.ones(big_bs)
#     tens_big = torch.zeros(tens_ones.shape[0], 2, 2)
#     tens_big[:,0,0] = tens_ones*2
#     tens_big[:,0,1] = tens_ones*0
#     tens_big[:,1,0] = tens_ones*0
#     tens_big[:,1,1] = tens_ones*4
#     inverse_big = torch.inverse(tens_big) 
#     print(inverse_big) # as expected
    
#     D_tens_test = D(1, 1, 2, 2, tens_ones)
#     print(D_tens_test)
#     print(D_tens_test.shape)
    
#     torch.t(torch.ones(3,1)).mm(torch.ones(3,1)*2)
#     a = torch.tensor([[1.0],[2.0]]).expand()
#     inv = torch.inverse(D_tens_test)
#     res = inv.mm(a) 
        
# def h_vec(L1, L2, M2, dt1, dt2, t2):
#     '''Coriolis and centripetal vector, a length-2 column vector.
#     function of static params and instantaneus angular velocities and elbow angle
#     Args: 
#         L1: length of upper arm. scalar quantity or torch.nn.Parameter
#         L2: length of fore arm. scalar quantity or torch.nn.Parameter
#         M2: mass of fore arm. scalar quantity or torch.nn.Parameter
#         dt1: current shoulder angular velocity. torch.Tensor(batch_size * num_particles)
#         dt2: current elbow angular velocity. torch.Tensor(batch_size * num_particles)
#         t2: current elbow angle. torch.Tensor(batch_size * num_particles)
#     Returns:
#         torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
#         dot product later; should be a length-2 column vector.
#         '''
#     assert(dt1.shape[0] == dt2.shape[0] & dt2.shape[0] == t2.shape[0])
#     # note: we are always assuming batched processing.
    
#     h_vec_tensor = torch.zeros(t2.shape[0], 2, 1) # output shape
#     h_vec_tensor[:,0,0] = -L1*L2*M2*dt2*(2*dt1 + dt2)*torch.sin(t2)/2
#     h_vec_tensor[:,1,0] = L1*L2*M2*dt1**2*torch.sin(t2)/2
    
#     return h_vec_tensor

#     # return torch.Tensor([-L1*L2*M2*dt2*(2*dt1 + dt2)*torch.sin(t2)/2,
#     #                L1*L2*M2*dt1**2*torch.sin(t2)/2])

# if test_args:
#     res = h_vec(1, 1, 0.5, torch.tensor(0.6), torch.tensor(0.2), torch.tensor(0.3))
#     res.shape
#     a = torch.ones(30)
#     b = torch.ones_like(a)*2
#     #b = torch.ones(31) # should fail assertion
#     c = torch.ones_like(a)*3
#     res = h_vec(1, 1, 0.5, a, b, c)
#     res.shape
    
# def c_vec(L1, L2, M1, M2, g, t1, t2):
#     '''gravity vector, a length-2 column vector. 
#     function of all static parameters, current configuration of both angles 
#     and gravity constant.
#     Args: 
#         L1: length of upper arm. scalar quantity or torch.nn.Parameter
#         L2: length of fore arm. scalar quantity or torch.nn.Parameter
#         M1: mass of upper arm. scalar quantity or torch.nn.Parameter
#         M2: mass of fore arm. scalar quantity or torch.nn.Parameter
#         t1: current shoulder angle. torch.Tensor(batch_size * num_particles)
#         t2: current elbow angle. torch.Tensor(batch_size * num_particles)
#     Returns:
#         torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
#         dot product later; should be a length-2 column vector.
#         '''
#     assert(t1.shape[0] == t2.shape[0])
#     # note: we are always assuming batched processing.

#     c_vec_tensor = torch.zeros(t2.shape[0], 2, 1) # output shape
#     c_vec_tensor[:,0,0] = L1*M1*g*torch.cos(t1)/2 + M2*g*(2*L1*torch.cos(t1) + 
#                                                           L2*torch.cos(t1 + t2))/2
#     c_vec_tensor[:,1,0] = L2*M2*g*torch.cos(t1 + t2)/2
    
#     return c_vec_tensor
    
#     # return torch.Tensor([L1*M1*g*torch.cos(t1)/2 + M2*g*(2*L1*torch.cos(t1) + L2*torch.cos(t1 + t2))/2, 
#     #           L2*M2*g*torch.cos(t1 + t2)/2])
# if test_args:
#     a = torch.ones(30)
#     b = torch.ones_like(a)*2
#     #b = torch.ones(31) # should fail assertion
#     res = c_vec(1, 1.2, 0.5, 0.7, 1, a, b)
#     res.shape

# # it will be easiest if newton_second_law is a method inside
# # emission. in that way, if we log theta, omega, and we have the
# # static params, no need for all these inputs?
# # I imagine not inputing the functions but rather applying them
# # outside the function.
# # so that the function sees the output of D, h_vec and c_vec

# def newton_second_law(latent_torques, theta_mat, omega_mat,
#                       D, h_vec, c_vec, L1, L2, 
#                       M1, M2, g, t1, t2):
#     '''compute instantaneous angular acceleration, a length-2 column vector, 
#     according to Newton's second law: 
#         force = mass*acceleration, 
#     which in the angular setting (and in 2D), becomes: 
#         torque (2X1) = intertia_tens (2X2) * angular_accel (2X1)
#     Here, we are multiplying both sides of the equation by 
#     intertia_tens**(-1) to remain with an expression for angular acceleration.
#     We are also taking into account "fictitious forces" coriolis/centripetal and gravity
#     forces. 
#     The function executes three operations:
#         (1) compute the inertia tensor and invert it, 
#         (2) subtract "fictitious forces" from torque vector
#         (3) compute the dot product between (1) and (2)
    
#     Args: 
        
#         D: a function defined above, that computes the inertia tensor from 
#             static parameters and angles.
#             Args:
#                 L1: length of upper arm. scalar quantity or torch.nn.Parameter
#                 L2: length of fore arm. scalar quantity or torch.nn.Parameter
#                 M1: mass of upper arm. scalar quantity or torch.nn.Parameter
#                 M2: mass of fore arm. scalar quantity or torch.nn.Parameter
#                 t2: current elbow angle. torch.Tensor(batch_size * num_particles)
#             Returns:
#                 torch.tensor [batch_size*num_particles, 2, 2]. These dimensions are
#                 important [batch_size * num_particles, [squared mat]] 
#                 because they allow us to later invert D
        
#         h_vec: a function defined above, computes Coriolis and centripetal vector. 
#             a length-2 column vector, which is a function of static params and 
#             instantaneus angular velocities and elbow angle.
#             Args: 
#                 L1: length of upper arm. scalar quantity or torch.nn.Parameter
#                 L2: length of fore arm. scalar quantity or torch.nn.Parameter
#                 M2: mass of fore arm. scalar quantity or torch.nn.Parameter
#                 dt1: current shoulder angular velocity. torch.Tensor(batch_size * num_particles)
#                 dt2: current elbow angular velocity. torch.Tensor(batch_size * num_particles)
#                 t2: current elbow angle. torch.Tensor(batch_size * num_particles)
#             Returns:
#                 torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
#                 dot product later; should be a length-2 column vector.
        
#         c_vec: gravity vector, a length-2 column vector. 
#         function of all static parameters, current configuration of both angles 
#         and gravity constant.
#         Args: 
#             L1: length of upper arm. scalar quantity or torch.nn.Parameter
#             L2: length of fore arm. scalar quantity or torch.nn.Parameter
#             M1: mass of upper arm. scalar quantity or torch.nn.Parameter
#             M2: mass of fore arm. scalar quantity or torch.nn.Parameter
#             t1: current shoulder angle. torch.Tensor(batch_size * num_particles)
#             t2: current elbow angle. torch.Tensor(batch_size * num_particles)
#         Returns:
#             torch.Tensor([batch_size * num_particles, 2, 1]). Size is important for
#             dot product later; should be a length-2 column vector.
        
                        
#     Returns:
#         torch.Tensor([batch_size * num_particles, 2, 1]). Should be a length-2 column vector,
#         if we have two angles, matching the angular velocity and angle vectors.
        
#         '''
#     assert(t1.shape[0] == t2.shape[0])
#     # note: we are always assuming batched processing.

#     c_vec_tensor = torch.zeros(t2.shape[0], 2, 1) # output shape
#     c_vec_tensor[:,0,0] = L1*M1*g*torch.cos(t1)/2 + M2*g*(2*L1*torch.cos(t1) + 
#                                                           L2*torch.cos(t1 + t2))/2
#     c_vec_tensor[:,1,0] = L2*M2*g*torch.cos(t1 + t2)/2
    
#     return c_vec_tensor
# # compute_prev_accel
    
# # next step -- define the emission model. 
# # two things to consider -- inits could be learned; not sure how inits are changing bw batches
# # potentially: inits define a shared distribution across batches, but we take
# # samples from that distribution and these samples will recieve different weights
#     # for different batches.
# # second, recall that we are running the functions in [batch_size*num_particles,...]
# # mode, so that before pusing things through coords we have to change view again. 
    
# below, see a forward method that runs but doesn't infer well 02/06/2020
# maybe the method is fine but we have some other issue with the dimensions
    # def forward(self, latents = None, time = None, previous_observations = None):
        
    #     # time 0
    #     # use initial angles to explain data
    #     # one could optimize these angles and velocities to start from a good initial point
    #     #print("latent: " , latents[-1].shape)
    #     if time == 0:
    #         # set lists at the first time step and then append values
    #         self.angles = [] # append inits
    #         self.velocity = [] # append inits. ToDo: for simplicity set at rest
    #         self.acceleration = [] # append just in the next time step
            
    #         # compute dimensions 
    #         self.batch_size = latents[-1].shape[0]
    #         self.num_particles = latents[-1].shape[1]
    #         self.dim_latents = latents[-1].shape[2]
            
    #         # append inits to dicts (no reshaping)
    #         #self.angles.append(self.init_angles) # previous version, more flexible, didnt work
    #         self.angles.append(torch.zeros_like(latents[-1]).view(-1,2))
    #         # self.angles.append(self.init_angles.view(
    #         #     self.batch_size, self.num_particles, self.dim_latents))
    #         # self.velocity.append(self.init_velocity) # previous version, more flexible, didnt work
    #         self.velocity.append(torch.zeros_like(latents[-1]).view(-1,2))

        
    #     else: # time not zero
    #         # accel_prev = Newton_2nd(torques, computed_D, computed_c_vec, computed_h_vec)
    #         # ToDo: check if they have in aesmc a better way to squeeze
    #         # compute previous acceleration based on previous torques, angles and velocity.
    #         # note: using latents[-2] because I assume that since then, 
    #         # we drew the current set of latents in the proposal.
    #         print('regular latents: ', latents[-2].shape)
    #         print('contig latents: ',latents[-2].contiguous().shape)
    #         print((latents[-2] == latents[-2].contiguous()).detach().numpy().all())
    #         # ToDo: the part below needs extra caution:
    #         accel_prev = self.Newton_2nd(
    #             latents[-2].contiguous().view( # squeezing. ToDo: not sure about line.
    #                                     self.batch_size*self.num_particles, 2, 1), # self.batch_size*self.num_particles ,2, 1),
    #                                     self.D(self.angles[-1][:,1]), 
    #                                     self.h_vec(self.velocity[-1][:,0], 
    #                                                self.velocity[-1][:,1], 
    #                                                self.angles[-1][:,1]), 
    #                                     self.c_vec(self.angles[-1][:,0], 
    #                                                self.angles[-1][:,1]))
    #         # append acceleration associated with previous time step.
    #         self.acceleration.append(accel_prev) # at time 1 we first append
    #         # compute current velocity based on previous velocity and accel.
    #         velocity_curr = self.Euler_step(self.velocity[-1], self.acceleration[-1])
    #         # compute current angles based on previous angles and velocity.
    #         angles_curr = self.Euler_step(self.angles[-1], self.velocity[-1])
    #         # append current iteration velocity and angles
    #         self.velocity.append(velocity_curr)
    #         self.angles.append(angles_curr)
          
    #     # for every timestep we conclude with forward kinematics given current angles
        
    #     #print("angles: " , self.angles[-1].shape)
    #         # careful of contiguous!
    #     mean_tensor = self.FW_kin_2D(self.angles[-1].contiguous().view(
    #              self.batch_size, self.num_particles, self.dim_latents))
        
    #     #print("mean of dist: " , mean_tensor.shape)
            
    #     return aesmc.state.set_batch_shape_mode(
    #     torch.distributions.Normal(mean_tensor, self.scale),
    #     aesmc.state.BatchShapeMode.FULLY_EXPANDED)
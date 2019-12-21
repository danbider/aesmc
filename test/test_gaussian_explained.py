# playing around with the gaussian model 
# we are at .../aesmc/test
import aesmc.train as train
import aesmc.losses as losses
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import unittest
from pprint import pprint
from models import gaussian # distribution classes are defined there


# define parameters
prior_std = 1

true_prior_mean = 0
true_obs_std = 1

prior_mean_init = 2
obs_std_init = 0.5

q_init_mult, q_init_bias, q_init_std = 2, 2, 2
q_true_mult, q_true_bias, q_true_std = gaussian.get_proposal_params(
    true_prior_mean, prior_std, true_obs_std) # check how you get that

# true_prior and true_likelihood are used just to generate data
# we initialize the instances, and they have a forward method
# that returns torch.distributions.Normal with specified params
true_prior = gaussian.Prior(true_prior_mean, prior_std)
pprint(vars(true_prior))

true_likelihood = gaussian.Likelihood(true_obs_std)
pprint(vars(true_likelihood))

num_particles = 2
batch_size = 10
num_iterations = 2000

'''the training stats object has a __call__ method 
that will be the callback for train.train'''
training_stats = gaussian.TrainingStats(logging_interval=500)

print('\nTraining the \"gaussian\" autoencoder.')
# initialize instances for distributions you want to learn
prior = gaussian.Prior(prior_mean_init, prior_std)
likelihood = gaussian.Likelihood(obs_std_init) 
inference_network = gaussian.InferenceNetwork(
    q_init_mult, q_init_bias, q_init_std)

# understand instances
pprint(vars(prior))
pprint(vars(likelihood)) # note: log_std = np.log(obs_std_init=0.5)
pprint(vars(inference_network)) # note: log_std = np.log(q_init_std=2)

#%% understand the data loading (don't need to run)

'''let's try to understand the data loading.'''
'''a Map-style dataset for torch.utils.data. we won't use
it below but just to illustrate that it's a method to sample data.
implemented by our class train.SyntheticDataset(torch.utils.data.Dataset)
Things to notice in the model - no transition, and also just 1 timepoint'''
dataset = train.SyntheticDataset(true_prior,None, true_likelihood,1, batch_size)
pprint(vars(dataset))

'''create a dataloader object (redefine dataset inside this func)'''
dataloader = train.get_synthetic_dataloader(
                true_prior, None, true_likelihood, 1, batch_size)                              
pprint(vars(dataloader))

'''inspect how the data are read inside train.train
note that unless we have the if statement, this loop
will go on forever. also , we print indices and tensors. 
each tensor has batch_size elements in it'''
for epoch_iteration_idx, observations in enumerate(dataloader):
    if epoch_iteration_idx == num_iterations:
                    break
    print(epoch_iteration_idx)
    print(observations)

#%% train and visualize
'''note that during training, the values of the parameters
of the pytorch.distributions instances that we defined are
constantly changing'''
train.train(dataloader=train.get_synthetic_dataloader(
                true_prior, None, true_likelihood, 1, batch_size),
            num_particles=num_particles,
            algorithm='iwae', # note the algorithm.
            initial=prior,
            transition=None,
            emission=likelihood,
            proposal=inference_network,
            num_epochs=1,
            num_iterations_per_epoch=num_iterations,
            optimizer_algorithm=torch.optim.SGD,
            optimizer_kwargs={'lr': 0.01},
            callback=training_stats)

plt.plot(training_stats.loss_history)

fig, axs = plt.subplots(5, 1, sharex=True, sharey=True)
fig.set_size_inches(10, 8)

mean = training_stats.prior_mean_history
obs = training_stats.obs_std_history
mult = training_stats.q_mult_history
bias = training_stats.q_bias_history
std = training_stats.q_std_history
data = [mean] + [obs] + [mult] + [bias] + [std]
true = [true_prior_mean, true_obs_std, q_true_mult, q_true_bias,
        q_true_std]

''' just to demonstrate how the instances in our workspace are changing
 in fact our callback function Trainingstats.__call__
 just looks at the instances and logs their values into 
 Trainingstats object'''
print(prior.mean.item() == mean[-1])

for ax, data_, true_, ylabel in zip(
    axs, data, true, ['$\mu_0$', '$\sigma$', '$a$', '$b$', '$c$']
):
    ax.plot(training_stats.iteration_idx_history, data_)
    ax.axhline(true_, color='black')
    ax.set_ylabel(ylabel)
    #  self.assertAlmostEqual(data[-1], true, delta=1e-1)

axs[-1].set_xlabel('Iteration')
fig.tight_layout()

filename = 'test_autoencoder_plots/gaussian.pdf'
fig.savefig(filename, bbox_inches='tight')
print('\nPlot saved to {}'.format(filename))
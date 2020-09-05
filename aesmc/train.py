# Dan 02/26 - we want a local import, not from the package.
from . import losses
from . import statistics


import itertools
import sys
import torch.nn as nn
import torch.utils.data


def get_chained_params(*objects):
    result = []
    for object in objects:
        if (object is not None) and isinstance(object, nn.Module):
            result = itertools.chain(result, object.parameters())

    if isinstance(result, list):
        return None
    else:
        return result

def train_tracking_data(dataloader,
                        num_particles,
                        algorithm,
                        model,
                        num_epochs,
                        num_iterations_per_epoch=None,
                        parameters = None,
                        optimizer_algorithm=torch.optim.Adam,
                        optimizer_kwargs={},
                        callback=None):
    # parameters = get_chained_params(model["initial"], model["transition"],
    #                                 model["emission"], model["proposal"])
    # parameters = get_chained_params(model["initial"], model["transition"],
    #                                 model["emission"], model["proposal"])
    print(list(parameters))
    #print('Training %i parameter groups' % len(list(parameters)))
    optimizer = optimizer_algorithm(parameters, **optimizer_kwargs)
    for epoch_idx in range(num_epochs):
        #torch.manual_seed(0) # to make sure data loader repeats batches across epochs.
        for epoch_iteration_idx, (observations, t_ind_start) in enumerate(dataloader):
            if torch.sum(torch.isnan(torch.cat(observations))) > 0:
                continue
            if num_iterations_per_epoch is not None: # for simulated data.
                if epoch_iteration_idx == num_iterations_per_epoch:
                    break
            optimizer.zero_grad()
            loss = losses.get_loss(observations, num_particles, algorithm,
                                  model["initial"], model["transition"],
                                    model["emission"], model["proposal"])
            loss.backward()
            optimizer.step()

            if callback is not None:
                callback(epoch_idx, epoch_iteration_idx, loss, model["initial"], model["transition"],
                                    model["emission"], model["proposal"])

def train(dataloader, num_particles, algorithm, initial, transition, emission,
          proposal, num_epochs, num_iterations_per_epoch=None,
          optimizer_algorithm=torch.optim.Adam, optimizer_kwargs={},
          callback=None):
    parameters = get_chained_params(initial, transition, emission, proposal)
    optimizer = optimizer_algorithm(parameters, **optimizer_kwargs)
    for epoch_idx in range(num_epochs):
        #torch.manual_seed(0) # to make sure data loader repeats batches across epochs.
        for epoch_iteration_idx, observations in enumerate(dataloader):
            if torch.sum(torch.isnan(torch.cat(observations)))>0:
                continue
            if num_iterations_per_epoch is not None:
                if epoch_iteration_idx == num_iterations_per_epoch:
                    break
            optimizer.zero_grad()
            loss = losses.get_loss(observations, num_particles, algorithm,
                                   initial, transition, emission, proposal)
            loss.backward()
            optimizer.step()

            if callback is not None:
                callback(epoch_idx, epoch_iteration_idx, loss, initial,
                         transition, emission, proposal)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, initial, transition, emission, num_timesteps,
                 batch_size):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size

    def __getitem__(self, index):
        # TODO this is wrong, obs can be dict
        return list(map(
            lambda observation: observation.detach().squeeze(0),
            statistics.sample_from_prior(self.initial, self.transition,
                                         self.emission, self.num_timesteps,
                                         self.batch_size)[1]))

    def __len__(self):
        return sys.maxsize  # effectively infinite


def get_synthetic_dataloader(initial, transition, emission, num_timesteps,
                             batch_size):
    return torch.utils.data.DataLoader(
        SyntheticDataset(initial, transition, emission, num_timesteps,
                         batch_size),
        batch_size=1,
        collate_fn=lambda x: x[0])

# Dan 02/26 - we want a local import, not from the package.
from . import inference
from . import math
from . import state

import torch

def empirical_expectation(value, log_weight, f):
    """Empirical expectation.

    Args:
        value: torch.Tensor
            [batch_size, num_particles, value_dim_1, ..., value_dim_N] (or
            [batch_size, num_particles])
        log_weight: torch.Tensor [batch_size, num_particles]
        f: function which takes torch.Tensor
            [batch_size, value_dim_1, ..., value_dim_N] (or
            [batch_size]) and returns a torch.Tensor
            [batch_size, dim_1, ..., dim_M] (or [batch_size])

    Returns: empirical expectation torch.Tensor
        [batch_size, dim_1, ..., dim_M] (or [batch_size])
    """

    assert(value.size()[:2] == log_weight.size())
    normalized_weights = math.exponentiate_and_normalize(log_weight, dim=1)

    # first particle
    f_temp = f(value[:, 0])
    w_temp = normalized_weights[:, 0]
    for i in range(f_temp.dim() - 1):
        w_temp.unsqueeze_(-1)

    emp_exp = w_temp.expand_as(f_temp) * f_temp

    # next particles
    for p in range(1, normalized_weights.size(1)):
        f_temp = f(value[:, p])
        w_temp = normalized_weights[:, p]
        for i in range(f_temp.dim() - 1):
            w_temp.unsqueeze_(-1)

        emp_exp += w_temp.expand_as(f_temp) * f_temp

    return emp_exp


def empirical_mean(value, log_weight):
    """Empirical mean.

    Args:
        value: torch.Tensor
            [batch_size, num_particles, dim_1, ..., dim_N] (or
            [batch_size, num_particles])
        log_weight: torch.Tensor [batch_size, num_particles]

    Returns: empirical mean torch.Tensor
        [batch_size, dim_1, ..., dim_N] (or [batch_size])
    """

    return empirical_expectation(value, log_weight, lambda x: x)


def empirical_variance(value, log_weight):
    """Empirical variance.

    Args:
        value: torch.Tensor
            [batch_size, num_particles, dim_1, ..., dim_N] (or
            [batch_size, num_particles])
        log_weight: torch.Tensor [batch_size, num_particles]
    Returns: empirical mean torch.Tensor
        [batch_size, dim_1, ..., dim_N] (or [batch_size])
    """

    return empirical_expectation(value, log_weight, lambda x: x**2) -\
        empirical_mean(value, log_weight)**2


def log_ess(log_weight):
    """Log of Effective sample size.

    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, num_particles] (or [num_particles])

    Returns: log of effective sample size [batch_size] (or [1])
    """
    dim = 1 if log_weight.ndimension() == 2 else 0

    return 2 * torch.logsumexp(log_weight, dim=dim) - \
        torch.logsumexp(2 * log_weight, dim=dim)


def ess(log_weight):
    """Effective sample size.

    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, num_particles] (or [num_particles])

    Returns: effective sample size [batch_size] (or [1])
    """

    return torch.exp(log_ess(log_weight))


# TODO: test
def sample_from_prior(initial, transition, emission, num_timesteps,
                      batch_size, repeat_data=False):
    """Samples latents and observations from prior

    Args:
        initial: a callable object (function or nn.Module) which has no
            arguments and returns a torch.distributions.Distribution or a dict
            thereof
        transition: a callable object (function or nn.Module) with signature:
            Args:
                previous_latents: list of length time where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int (zero-indexed)
                previous_observations: list of length time where each element
                    is a tensor [batch_size, ...] or a dict thereof
            Returns: torch.distributions.Distribution or a dict thereof
        emission: a callable object (function or nn.Module) with signature:
            Args:
                latents: list of length (time + 1) where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int (zero-indexed)
                previous_observations: list of length time where each element
                    is a tensor [batch_size, ...] or a dict thereof
            Returns: torch.distributions.Distribution or a dict thereof
        num_timesteps: int
        batch_size: int
        repeat_data: bool, Dan added this to repeat data across batches.

    Returns:
        latents: list of tensors (or dict thereof) [batch_size] of length
            len(observations)
        observations: list of tensors [batch_size, ...] or dicts thereof
    """

    latents = []
    observations = []
    
    if repeat_data:
        print('repeating data in batch.')
        for time in range(num_timesteps):
            if time == 0:
                latents.append(state.sample(initial(), 1, 1)) # .expand(
                    # batch_size, 1, -1) # immediately after state sample
            else:
                latents.append(state.sample(transition(
                    previous_latents=latents, time=time,
                    previous_observations=observations[:time]), 1, 1))
            observations.append(state.sample(emission(
                latents=latents, time=time,
                previous_observations=observations[:time]), 1, 1))
        for i in range(len(latents)):
            latents[i] = latents[i].expand(batch_size, 1, -1)
            observations[i] = observations[i].expand(batch_size, 1, -1)
    else: 
        print('each batch contains different data.')
        for time in range(num_timesteps):
            if time == 0:
                latents.append(state.sample(initial(), batch_size, 1))
            else:
                latents.append(state.sample(transition(
                    previous_latents=latents, time=time,
                    previous_observations=observations[:time]), batch_size, 1))
            observations.append(state.sample(emission(
                latents=latents, time=time,
                previous_observations=observations[:time]), batch_size, 1))

    def squeeze_num_particles(value):
        if isinstance(value, dict):
            return {k: squeeze_num_particles(v) for k, v in value.items()}
        else:
            return value.squeeze(1)

    return tuple(map(lambda values: list(map(squeeze_num_particles, values)),
                 [latents, observations]))

def sim_data_from_model(model_dict, num_timesteps, batch_size, repeat_data):
    '''this utility just allows to pass a model_dict with
    "initial", "transition", and "emission" objects.'''
    sim_lats, sim_observs = sample_from_prior(
        model_dict["initial"], model_dict["transition"],
        model_dict["emission"], num_timesteps, batch_size, repeat_data)
    return sim_lats, sim_observs

def compute_log_joint(full_model, latents, observations):
    '''
    loop over time points and compute p(x,y)
    Args: 
        full_model: dict with keys "initial", "transition", and "emission" which are torch.Distribution objects
        latents: list of len num_timesteps with torch.size(batch_size, dim_latent). no particle dim.
        observations: list of len num_timesteps with torch.size(batch_size, dim_observs). no particle dim.
    Returns: scalar, log joint, for each element in the batch dimension, so torch.size(batch_size)
    '''
    inst_log_joint = []
    for i in range(len(observations)):
        if i == 0:
            log_trans = full_model["initial"]().log_prob(
                latents[0].unsqueeze(1))
        else:
            log_trans = full_model["transition"](
                previous_latents=[latents[i - 1].unsqueeze(1)]).log_prob(
                    latents[i].unsqueeze(1))
        log_emission = full_model["emission"](
            latents=[latents[i].unsqueeze(1)]).log_prob(
                observations[i].unsqueeze(1))

        inst_log_joint.append(
            log_trans + log_emission
        )  # eventually, list with len num_timesteps, each element torch.size([batch_size,1])

    # list of tensors -> tensor of torch.size([num_timesteps, batch_size])
    temp = torch.stack(inst_log_joint,
                       dim=0).squeeze(-1)  # get rid of the particle dim
    # sum across num_timesteps to get tensor of torch.Size([batch_size])
    log_joint = torch.sum(temp, dim=0)

    return log_joint
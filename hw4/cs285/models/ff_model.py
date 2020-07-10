from cs285.infrastructure.utils import normalize, unnormalize, MLP
import numpy as np
import torch
from torch import nn

class FFModel:
    def __init__(self, ac_dim, ob_dim, n_layers, size, device, learning_rate = 0.001):
        # init vars
        self.device = device

        #TODO - specify ouput dim and input dim of delta func MLP
        self.delta_func = MLP(input_dim = TODO,
                              output_dim = TODO,
                              n_layers = n_layers,
                              size = size,
                              device = self.device,
                              discrete = True)

        #TODO - define the delta func optimizer. Adam optimizer will work well.
        self.optimizer = TODO

    #############################

    def get_prediction(self, obs, acs, data_statistics):
        if len(obs.shape) == 1 or len(acs.shape) == 1:
            obs = np.squeeze(obs)[None]
            acs = np.squeeze(acs)[None]

        # TODO(Q1) normalize the obs and acs above using the normalize function and data_statistics
        norm_obs = TODO
        norm_acs = TODO

        norm_input = torch.Tensor(np.concatenate((norm_obs, norm_acs), axis = 1)).to(self.device)
        norm_delta = self.delta_func(norm_input).cpu().detach().numpy()

        # TODO(Q1) Unnormalize the the norm_delta above using the unnormalize function and data_statistics
        delta = TODO
        # TODO(Q1) Return the predited next observation (You will use obs and delta)
        return TODO

    def update(self, observations, actions, next_observations, data_statistics):
        # TODO(Q1) normalize the obs and acs above using the normalize function and data_statistics (same as above)
        norm_obs = TODO
        norm_acs = TODO

        pred_delta = self.delta_func(torch.Tensor(np.concatenate((norm_obs, norm_acs), axis = 1)).to(self.device))
        # TODO(Q1) Define a normalized true_delta using observations, next_observations and the delta stats from data_statistics
        true_delta = TODO

        # TODO(Q1) Define a loss function that takes as input normalized versions of predicted change in state and true change in state
        loss = TODO
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

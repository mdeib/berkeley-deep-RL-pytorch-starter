import numpy as np
import torch
import torch.nn as nn

class MLPPolicy(nn.Module):

    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        device,
        lr = 1e-4,
        training=True,
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__()

        # init vars
        self.training = training
        self.device = device

        # network architecture
        #TODO -build the network architecture
        #HINT -build an nn.Modulelist() using the passed in parameters

        #loss and optimizer
        if self.training:
            # TODO define the loss that will be used to train this policy
            self.loss_func = TODO
            self.optimizer = torch.optim.Adam(self.parameters(), lr)

        self.to(device)

    ##################################

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore(self, filepath):
        self.load_state_dict(torch.load(filepath))

    ##################################

    # query this policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        return TODO

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):

    """
        This class is a special case of MLPPolicy,
        which is trained using supervised learning.
        The relevant functions to define are included below.
    """

    def update(self, observations, actions):
        assert self.training, 'Policy must be created with training = true in order to perform training updates...'

        # TODO define network update
        #HINT - you need to calculate the prediction loss and then use optimizer.step()

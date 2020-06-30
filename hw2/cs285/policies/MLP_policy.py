import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        device,
        discrete,
        activation = nn.Tanh()):
        super().__init__()

        self.discrete = discrete

        #TODO -build the network architecture -can be taken from HW1
        #HINT -build an nn.Modulelist() using the passed in parameters

        #if continuous define logstd variable
        if not self.discrete:
            self.logstd = nn.Parameter(torch.zeros(ac_dim))

        self.to(device)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        if self.discrete:
            return x
        else:
            return (x, self.logstd.exp())

class MLPPolicy:
    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        device,
        learning_rate,
        training=True,
        discrete=False,
        nn_baseline=False,
        **kwargs):
        super().__init__()

        # init vars
        self.device = device
        self.discrete = discrete
        self.training = training
        self.nn_baseline = nn_baseline

        # network architecture
        self.policy_mlp = MLP(ac_dim, ob_dim, n_layers, size, device, discrete)
        params = list(self.policy_mlp.parameters())
        if self.nn_baseline:
            self.baseline_mlp = MLP(1, ob_dim, n_layers, size, device, True)
            params += list(self.baseline_mlp.parameters())

        #optimizer
        if self.training:
            self.optimizer = torch.optim.Adam(params, lr = learning_rate)

    ##################################

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

    # query the neural net that's our 'policy' function, as defined by the policy_mlp above
    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        raise NotImplementedError
        #implement similar to HW1

    def get_log_prob(self, network_outputs, actions_taken):
        actions_taken = torch.Tensor(actions_taken).to(self.device)
        if self.discrete:
            #log probability under a categorical distribution
            network_outputs = nn.functional.log_softmax(network_outputs).exp()
            return torch.distributions.Categorical(network_outputs).log_prob(actions_taken)
        else:
            #log probability under a multivariate gaussian
            return torch.distributions.Normal(network_outputs[0], network_outputs[1]).log_prob(actions_taken).sum(-1)

#####################################################
#####################################################

class MLPPolicyPG(MLPPolicy):

    def update(self, observations, acs_na, adv_n = None, acs_labels_na = None, qvals = None):
        policy_output = self.policy_mlp(torch.Tensor(observations).to(self.device))
        logprob_pi = self.get_log_prob(policy_output, acs_na)

        #TODO Don't forget to zero out the gradient

        # TODO: define the loss that should be optimized when training a policy with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: look at logprob_pi above
        # HINT3: don't forget that we need to MINIMIZE this self.loss
            # but the equation above is something that should be maximized
        #HINT4: don't forget to propagate the loss backward

        if self.nn_baseline:
            baseline_prediction = self.baseline_mlp(torch.Tensor(observations).to(self.device)).view(-1)
            baseline_target = torch.Tensor((qvals - qvals.mean()) / (qvals.std() + 1e-8)).to(self.device)

            # TODO: define the loss that should be optimized for training the baseline
            # HINT1: use nn.functional.mse_loss, similar to SL loss from hw1
            # HINT2: we want predictions (self.baseline_prediction) to be as close as possible to the labels (self.targets_n)
            # HINT3: don't forget to propagate the loss backward

        #step the optimizer
        return loss

#####################################################
#####################################################

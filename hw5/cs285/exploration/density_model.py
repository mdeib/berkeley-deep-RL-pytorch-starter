import numpy as np
import torch
from torch import nn
from cs285.infrastructure.utils import MLP

class Histogram:
    def __init__(self, nbins, preprocessor):
        self.nbins = nbins
        self.total = 0.
        self.hist = {}
        for i in range(int(self.nbins)):
            self.hist[i] = 0
        self.preprocessor = preprocessor

    def update_count(self, state, increment):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE
            args:
                state: numpy array
                increment: int
            TODO:
                1. increment the entry "bin_name" in self.hist by "increment"
                2. increment self.total by "increment"
        """
        bin_name = self.preprocessor(state)
        raise NotImplementedError

    def get_count(self, states):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE
            args:
                states: numpy array (bsize, ob_dim)
            returns:
                counts: numpy_array (bsize)
            TODO:
                For each state in states:
                    1. get the bin_name using self.preprocessor
                    2. get the value of self.hist with key bin_name
        """
        raise NotImplementedError
        return counts

    def get_prob(self, states):
        """
            ### PROBLEM 1
            ### YOUR CODE HERE
            args:
                states: numpy array (bsize, ob_dim)

            returns:
                return the probabilities of the state (bsize)
            NOTE:
                remember to normalize by float(self.total)
        """
        raise NotImplementedError
        return probs

class RBF:
    """
        https://en.wikipedia.org/wiki/Radial_basis_function_kernel
        https://en.wikipedia.org/wiki/Kernel_density_estimation
    """
    def __init__(self, sigma):
        self.sigma = sigma
        self.means = None

    def fit_data(self, data):
        """
            ### PROBLEM 2
            ### YOUR CODE HERE
            args:
                data: list of states of shape (ob_dim)
            TODO:
                We simply assign self.means to be equal to the data points.
                Let the length of the data be B
                self.means: np array (B, ob_dim)
        """
        B, ob_dim = len(data), len(data[0])
        raise NotImplementedError
        self.means = None
        assert self.means.shape == (B, ob_dim)

    def get_prob(self, states):
        """
            ### PROBLEM 2
            ### YOUR CODE HERE
            given:
                states: (b, ob_dim)
                    where b is the number of states we wish to get the
                    probability of
                self.means: (B, ob_dim)
                    where B is the number of states in the replay buffer
                    we will plop a Gaussian distribution on top of each
                    of self.means with a std of self.sigma
            TODO:
                1. Compute deltas: for each state in states, compute the
                    difference between that state and every mean in self.means.
                2. Euclidean distance: sum the squared deltas
                3. Gaussian: evaluate the probability of the state under the
                    gaussian centered around each mean. The hyperparameters
                    for the reference solution assume that you do not normalize
                    the gaussian. This is fine since the rewards will be
                    normalized later when we compute advantages anyways.
                4. Average: average the probabilities from each gaussian
        """
        b, ob_dim = states.shape
        if self.means is None:
            # Return a uniform distribution if we don't have samples in the
            # replay buffer yet.
            return (1.0/len(states))*np.ones(len(states))
        else:
            B, replay_dim = self.means.shape
            assert states.ndim == self.means.ndim and ob_dim == replay_dim

            # 1. Compute deltas
            deltas = raise NotImplementedError
            assert deltas.shape == (b, B, ob_dim)

            # 2. Euclidean distance
            euc_dists = raise NotImplementedError
            assert euc_dists.shape == (b, B)

            # Gaussian
            gaussians = raise NotImplementedError
            assert gaussians.shape == (b, B)

            # 4. Average
            densities = raise NotImplementedError
            assert densities.shape == (b,)

            return densities

class Exemplar(nn.Module):
    def __init__(self, ob_dim, hid_dim, learning_rate, kl_weight, device):
        super().__init__()
        self.ob_dim = ob_dim
        self.hid_dim = hid_dim
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.device = device

        '''
        TODO:
            define input and output size for the two encoders and the discriminator
            HINT: there should be self.hid_dim latent variables, half from each encoder
        '''

        self.encoder1 = MLP(input_dim = None,
                            output_dim = None,
                            n_layers = 2,
                            size = self.hid_dim,
                            device = self.device,
                            discrete = False)

        self.encoder2 = MLP(input_dim = None,
                            output_dim = None,
                            n_layers = 2,
                            size = self.hid_dim,
                            device = self.device,
                            discrete = False)

        self.discriminator = MLP(input_dim = None,
                                output_dim = None,
                                n_layers = 2,
                                size = self.hid_dim,
                                device = self.device,
                                discrete = True)
        '''
        TODO:
            prior_mean and prior_cov are for a standard normal distribution
            both have the same dimension as output dimension of the encoder network

            HINT1: Use torch.eye for the covariance matrix (Diagonal of covariance matrix are the variances)
            HINT2: Don't forget to add both to the correct device
        '''
        prior_means = None
        prior_cov = None
        self.prior = torch.distributions.MultivariateNormal(prior_means, prior_cov)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

    def forward(self, state1, state2):
        encoded1_mean, encoded1_std = self.encoder1(torch.Tensor(state1).to(self.device))
        encoded2_mean, encoded2_std = self.encoder2(torch.Tensor(state2).to(self.device))

        # sample epsilon1 and epsilon2 from the prior (don't forget to add then to self.device)
        epsilon1 = None
        epsilon2 = None

        #Do the reparameterization trick using the mean, std, and epsilon from above for each encoder output
        latent1 = None
        latent2 = None

        logit = self.discriminator(torch.cat([latent1, latent2], axis = 1)).squeeze()
        return logit

    def get_log_likelihood(self, state1, state2, target):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state1: np array (batch_size, ob_dim)
                state2: np array (batch_size, ob_dim)

            TODO:
                log likelihood of state1 == state2
                get logit by using the forward function
                discriminator_dist is a bernouli distribution made with the logit
                calculate the log_likelihood with .log_prob

            Hint:
                what should be the value of self.discrim_target?
        """
        logit = None
        discriminator_dist = None
        log_likelihood = None
        return log_likelihood

    def update(self, state1, state2, target):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE
            args:
                state1: np array (batch_size, ob_dim)
                state2: np array (batch_size, ob_dim)
                target: np array (batch_size, 1)
            TODO:
                train the density model and return
                    ll: log_likelihood
                    kl: kl divergence
                    elbo: elbo
        """
        assert state1.ndim == state2.ndim == target.ndim
        assert state1.shape[1] == state2.shape[1] == self.ob_dim
        assert state1.shape[0] == state2.shape[0] == target.shape[0]

        log_likelihood = self.get_log_likelihood(state1, state2, target)

        encoded1_mean, encoded1_std = self.encoder1(torch.Tensor(state1).to(self.device))
        encoded2_mean, encoded2_std = self.encoder2(torch.Tensor(state2).to(self.device))

        '''
        Use torch.distributions.MultivariateNormal to define the distributions for each part of the latent space
        HINT: the std must be made into a convariance matrix using torch.diag() (Don't forget to square std to get variance!)
        '''
        encoded1_dist = None
        encoded2_dist = None

        '''
        kl: shape: (batch_size): calculate kl1 and kl2 using torch.distributions.kl.kl_divergence
            to find kl divergence between each encoded_dist and self.prior
        '''
        kl1 = None
        kl2 = None
        kl = (kl1 + kl2)

        '''
        elbo: shape: scalar: subtract the kl (weighted by self.kl_weight) from the
              log_likelihood, and average over the batch
        '''
        elbo = (log_likelihood - (kl * self.kl_weight)).mean()

        #Use the optimizer to minimize -elbo (Note the negative sign!)
        raise NotImplementedError

        #map ll, kl, elbo to numpy arrays for logging
        ll, kl, elbo = map(lambda x: x.cpu().detach().numpy(), (log_likelihood, kl, elbo))
        return ll, kl, elbo

    def get_prob(self, state):
        """
            ### PROBLEM 3
            ### YOUR CODE HERE

            args:
                state: np array (batch_size, ob_dim)

            TODO:
                likelihood:
                    evaluate the discriminator D(x,x) on the same input
                prob:
                    compute the probability density of x from the discriminator
                    likelihood (see homework doc)
        """
        likelihood = None

        # avoid divide by 0 and log(0)
        likelihood = np.clip(np.squeeze(likelihood), 1e-5, 1-1e-5)
        prob = None
        return prob

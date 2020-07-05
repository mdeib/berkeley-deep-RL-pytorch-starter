from cs285.infrastructure.models import *
import torch
from torch import nn

class DQNCritic:
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.device = hparams['device']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = self.ob_dim
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['ac_dim']
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['grad_norm_clipping']
        self.gamma = hparams['gamma']

        self.optimizer_spec = optimizer_spec

        if self.env_name == 'LunarLander-v2':
            self.Q_func = LL_DQN(self.ac_dim, self.input_shape, self.device)
            self.target_Q_func = LL_DQN(self.ac_dim, self.input_shape, self.device)

        elif self.env_name == 'PongNoFrameskip-v4':
            self.Q_func = atari_DQN(self.ac_dim, self.input_shape, self.device)
            self.target_Q_func = atari_DQN(self.ac_dim, self.input_shape, self.device)

        else: raise NotImplementedError

        self.optimizer = self.optimizer_spec.constructor(self.Q_func.parameters(), lr = 1, **self.optimizer_spec.kwargs)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.optimizer_spec.lr_schedule)

    def get_loss(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        ob, ac, rew, next_ob, done = map(lambda x: torch.from_numpy(x).to(self.device), [ob_no, ac_na, re_n, next_ob_no, terminal_n])

        with torch.no_grad():
            if self.double_q:
                # You must fill this part for Q2 of the Q-learning potion of the homework.
                # In double Q-learning, the best action is selected using the Q-network that
                # is being updated, but the Q-value for this action is obtained from the
                # target Q-network. See page 5 of https://arxiv.org/pdf/1509.06461.pdf for more details.
                max_ac = TODO
            else:
                max_ac = TODO

        curr_Q = self.Q_func(ob).gather(-1, ac.long().view(-1, 1)).squeeze()
        # TODO calculate the optimal Qs for next_ob using max_ac
        # HINT1: similar to how it is done above
        best_next_Q = TODO
        # TODO calculate the targets for the Bellman error
        # HINT1: as you saw in lecture, this would be:
            #currentReward + self.gamma * best_next_Q * (1 - self.done_mask_ph)
        calc_Q = TODO

        return nn.functional.smooth_l1_loss(curr_Q, calc_Q) #Huber Loss


    def update(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        self.optimizer.zero_grad()

        loss = self.get_loss(ob_no, ac_na, re_n, next_ob_no, terminal_n)
        loss.backward()

        nn.utils.clip_grad_norm_(self.Q_func.parameters(), max_norm = self.grad_norm_clipping) #perform grad clipping
        self.optimizer.step() #take step with optimizer
        self.lr_scheduler.step() #move forward learning rate

        return loss

import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )
        self.ac_dim = ac_dim

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        # note here the obs is a one-dim array.
        obs = ptu.from_numpy(obs)
        action = self.forward(obs)
        # action dim is [action_dim]
        return ptu.to_numpy(action.sample())

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            return torch.distributions.categorical.Categorical(logits=self.logits_net(obs))
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            return torch.distributions.normal.Normal(
                loc=self.mean_net(obs), scale=torch.exp(self.logstd))
        return None

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        "The q values should be the 3rd input argment."
        all_state_num = obs.shape[0]
        if 'q_values_array' in kwargs:
            all_q_values = kwargs['q_values_array']
        else:
            all_q_values = np.ones(all_state_num)
        # Moves np array to tensor in GPU.
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        all_q_values = ptu.from_numpy(all_q_values)
        if self.discrete:
            logits = self.forward(obs)
            loss = torch.nn.functional.cross_entropy(
                logits, actions, weight=all_q_values)
        else:
            # We assume it's a gaussian distribution.
            normal_dist = torch.distributions.normal.Normal(
                loc=self.forward(obs), scale=torch.exp(self.logstd))
            loss = torch.mean(normal_dist.log_prob(actions) * all_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'training_loss': ptu.to_numpy(loss)}
        # raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        # currently, we treat advantages as q values for simply policy gradient.
        policy_distribution = self.forward(obs)
        loglikelihood = policy_distribution.log_prob(actions)
        loss = torch.neg(torch.mean(torch.mul(loglikelihood, advantages)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "Actor Loss": ptu.to_numpy(loss),
        }

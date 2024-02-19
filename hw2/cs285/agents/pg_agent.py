from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).

        Note: the sequence is not a tensor. In the outer call, it's a python list.
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.

        # Assemble the obs, action, q values into 3 big tensor, the dim is [num_traj * traj_size_T, ...]
        # Note the traj_size_T is not the same among all trajs.
        # Thus, we can use one batch run to conduct the policy gradient updates.
        all_obs = np.concatenate(obs, axis=0)
        all_actions = np.concatenate(actions, axis=0)
        all_q_values = np.concatenate(q_values, axis=0)
        all_rewards = np.concatenate(rewards, axis=0)
        all_terminals = np.concatenate(terminals, axis=0)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            all_obs, all_rewards, all_q_values, all_terminals
        )
        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        
        info: dict = self.actor.update(all_obs, all_actions, advantages=advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            for _ in range(self.baseline_gradient_steps):
                critic_info: dict = self.critic.update(all_obs, all_q_values)

            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""
        # one reward is a traj's R vectors. rewars are multiple trajs.
        # return a list, each value is a 1-D np array which contains the Q(s_t, a_t), the dim is t.
        # Each element of such list represents one traj's Q functions (i.e. a one-d array).
        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            # in this case, we are not really computing the Q value, because it's not defined as from current state to terminal.
            q_values = [self._discounted_return(reward) for reward in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = [self._discounted_reward_to_go(reward) for reward in rewards]

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        each input is 2-D array, whose dimensions are [Batch_size, traj length]
        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages? -> it's just Q functions
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            v_values = np.squeeze(ptu.to_numpy(self.critic.forward(ptu.from_numpy(obs))), axis=1)
            assert v_values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                # A is r_t + gamma * V_t+1 - V_t
                # We could use terminals to determine the end state
                # For the end state: A = r_t or here just the q_val_t
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size)
                for i in range(batch_size):
                    if terminals[i]:
                        advantages[i] = q_values[i]
                    else:
                        advantages[i] = rewards[i] + v_values[i + 1] * self.gamma - v_values[i]
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]
                # this is the delta_t array
                delta_t = np.zeros(batch_size)
                for i in range(batch_size):
                    if terminals[i]:
                        delta_t[i] = q_values[i]
                    else:
                        delta_t[i] = rewards[i] + v_values[i + 1] * self.gamma - v_values[i]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                delta_t = np.append(delta_t, [0])
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    if terminals[i] == 1:
                        advantages[i] = delta_t[i]
                    else:
                        advantages[i] = delta_t[i] + self.gamma * self.gae_lambda * advantages[i + 1]

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            advantages_mean = np.mean(advantages)
            advantages_std = np.std(advantages)
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        t_seq = np.arange(0, len(rewards), step=1.0, dtype=float)
        gamma_seq = np.power(self.gamma, t_seq)
        weighted_reward = np.sum(gamma_seq * rewards)
        res = np.ones(len(rewards)) * weighted_reward
        # print(res.shape)
        # print(res)
        return res


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        rewards should be a np array.
        """
        t_horizon = len(rewards)
        t_seq = np.arange(0, t_horizon, step=1.0, dtype=float)
        gamma_seq = np.power(self.gamma, t_seq)
        gamma_matrix = np.zeros(shape=[t_horizon, t_horizon])
        for row_id in range(t_horizon):
            gamma_matrix[row_id][row_id:t_horizon] = gamma_seq[0: t_horizon - row_id]
        res = np.matmul(gamma_matrix, rewards)
        # print(res)
        return res

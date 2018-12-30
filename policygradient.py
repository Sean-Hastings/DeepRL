import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import Network, DeepAgent
from time import time
import argparse


class PGAgent(DeepAgent):
    """
    Policy gradient based, pytorch neural net backed agent.

    This agent uses the gradient with respect to the policy to learn the policy.
    Because the policy space and its gradients can often be very noisy, this is a
    relatively high-variance approach.  To partially compensate for this it uses
    batches to reduce the variance.
    """

    def __init__(self, env, network, batch_size=1, alpha=.003, gamma=.99, use_cuda=True):
        DeepAgent.__init__(self, env, alpha, gamma, use_cuda)
        self.batch_size = batch_size

        # Network Prep
        device           = torch.device('cuda' if use_cuda else 'cpu')
        self.network     = network.to(device)
        self.opt         = optim.Adam(self.network.parameters(), lr=self.alpha)

        # Batch Prep
        self.sar           = []
        self.ep_counter    = 0
        self.batch_rewards = None
        self.batch_probs   = None


    def _decide(self, state):
        a_probs = self.network.forward(state)
        a_dist  = torch.distributions.Categorical(a_probs)
        return a_dist.sample()


    def process_move(self, state, action, reward, done, next_state):
        """
        Stores move for MonteCarlo evaluation when the episode ends.

        Uses an array of moves to track full episodes at a time, which it then
        records and, once it has enough recorded for a batch, updates its
        gradients on.
        """
        self.sar += [(state, action, reward)]
        if done:
            self.record_episode()
            self.ep_counter += 1
            if self.ep_counter >= self.batch_size:
                self.update_parameters()


    def record_episode(self):
        """Calculates and stores discounted rewards and log probabilities."""
        states, actions, episode_rewards = list(zip(*self.sar))

        discounted_episode_rewards = self.Tensor([0] * len(episode_rewards))
        cumulative                 = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = episode_rewards[i] + cumulative * self.gamma
            discounted_episode_rewards[i] = cumulative

        state_probs = self.network.forward(self.Tensor(states))
        state_dist  = torch.distributions.Categorical(probs=state_probs)
        log_probs   = state_dist.log_prob(self.Tensor(actions).squeeze())

        if self.batch_rewards is None:
            self.batch_rewards = discounted_episode_rewards.unsqueeze(1)
            self.batch_probs   = log_probs.unsqueeze(1)
        else:
            self.batch_rewards = torch.cat([self.batch_rewards, discounted_episode_rewards.unsqueeze(1)])
            self.batch_probs   = torch.cat([self.batch_probs,   log_probs.unsqueeze(1)])


    def update_parameters(self):
        """
        Uses stored batch to update gradients.

        The subtraction of the mean from the discounted rewards is a simple
        variance-reduction trick.
        """
        for param in self.network.parameters():
            param.grad = None

        self.batch_rewards = self.batch_rewards.squeeze()
        self.batch_probs   = self.batch_probs.squeeze()

        mean               = torch.mean(self.batch_rewards)
        self.batch_rewards = self.batch_rewards - mean

        value_error = torch.mean(-1 * self.batch_probs * self.batch_rewards)
        value_error.backward()
        self.opt.step()

        self.batch_probs   = None
        self.batch_rewards = None


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", help="Gym environment to train on", default='CartPole-v0')
    parser.add_argument("-hi", "--hidden_size", help="Hidden layer sizes, separated by whitespace", default='16')
    parser.add_argument("-t", "--training_episodes", help="Maximum number of episodes to train", default=100000)
    parser.add_argument("-a", "--alpha", help="Step size", type=float, default=.003)
    parser.add_argument("-g", "--goal_return", help="Goal return", type=float, default=195)
    parser.add_argument("-b", "--batch_size", help="Episodes per gradient update", type=int, default=100)
    args = parser.parse_args()

    args.hidden_size = [int(val) for val in args.hidden_size.split()]

    env = gym.make(args.environment)
    use_cuda = torch.cuda.is_available()

    state_shape  = list(env.reset().shape)
    state_shape  = [1] + state_shape[:-1]
    action_count = np.product(env.action_space.shape)
    network      = Network(state_shape, args.hidden_size, action_count)

    agent = PGAgent(env, network, batch_size=1, alpha=args.alpha, use_cuda=use_cuda)
    _time = time()
    converged = agent.train(args.training_episodes, goal_return=args.goal_return, smoothing_eps=100)
    if converged:
        print("Solved in %.1f minutes" % ((time() - _time) / 60.0))
    else:
        print("Failed to converge")

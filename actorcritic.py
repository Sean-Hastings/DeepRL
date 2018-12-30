import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import Network, ReplayMemory, DeepAgent
from time import time


class ACAgent(DeepAgent):
    """
    Actor-critic based, pytorch neural net backed agent.

    This agent uses policy-gradient methods to train a policy, but rather than
    train the policy directly on the (discounted) return it trains it on a
    neural network approximation of the return.  This approximation is in turn
    trained via experience replay on batches of individual, non-connected moves.
    """

    def __init__(self, env, policy_network, value_network, alpha=.003, gamma=.99, memory_size=10000, batch_size=64, use_cuda=True):
        DeepAgent.__init__(self, env, alpha, gamma, use_cuda)

        # Network prep
        device              = torch.device('cuda' if use_cuda else 'cpu')
        self.policy_network = policy_network.to(device)
        self.value_network  = value_network.to(device)
        self.policy_opt     = optim.Adam(self.policy_network.parameters(), lr=alpha)
        self.value_opt      = optim.Adam(self.value_network.parameters(), lr=alpha)

        # Experience replay prep
        self.memory         = ReplayMemory(max_size=memory_size)
        self.batch_size     = batch_size

    def _decide(self, state):
        a_probs = self.policy_network.forward(state)
        a_dist  = torch.distributions.Categorical(probs=a_probs)
        return a_dist.sample()

    def process_move(self, state, action, reward, done, next_state):
        self.memory.record((state, action, reward, not done, next_state))
        self.update_parameters()

    def update_parameters(self):
        batch = self.memory.recall(self.batch_size)

        if batch is not None:
            states, actions, rewards, dones, next_states = list(zip(*batch))
            t_states  = self.Tensor(states)
            t_actions = self.Tensor(actions)
            t_rewards = self.Tensor(rewards)
            t_done    = self.Tensor(dones)
            t_nstates = self.Tensor(next_states)

            nstate_values = None
            with torch.no_grad():
                nstate_values = self.value_network.forward(t_nstates).squeeze()

            state_values  = self.value_network.forward(t_states).squeeze()
            advantages    = t_rewards - state_values + self.gamma * nstate_values

            state_probs = self.policy_network.forward(t_states)
            state_dist  = torch.distributions.Categorical(probs=state_probs)
            log_probs   = state_dist.log_prob(t_actions)

            policy_loss = torch.mean(-1 * log_probs * advantages)
            self.policy_opt.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_opt.step()

            value_loss = F.mse_loss(state_values, t_rewards + t_done * self.gamma * nstate_values)
            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()


if  __name__ == '__main__':
    ENVIRONMENT   = 'Pong-ram-v0'
    CONV_LAYERS   = None#[(1, 8, 3, True), (8, 8, 3, True), (8, 8, 3, True), (8, 8, 3, True)]
    HIDDEN_LAYERS = [256]
    TRAIN_EPS     = 1000 * 1000 * 1000

    env      = gym.make(ENVIRONMENT)
    use_cuda = torch.cuda.is_available()

    state_shape    = list(env.reset().shape)
    state_shape    = [1] + state_shape[:-1]
    action_count   = np.product(env.action_space.shape)
    policy_network = Network(state_shape, HIDDEN_LAYERS, action_count, conv=CONV_LAYERS)
    value_network  = Network(state_shape, HIDDEN_LAYERS, 1, conv=CONV_LAYERS, softmax=False)

    agent = ACAgent(env,
                    policy_network,
                    value_network,
                    alpha=.003,
                    gamma=.99,
                    memory_size=10000,
                    batch_size=1024,
                    use_cuda=use_cuda)
    _time = time()
    agent.train(TRAIN_EPS, goal_return=15, smoothing_eps=1)
    print("Solved in %.1f minutes" % ((time() - _time) / 60.0))

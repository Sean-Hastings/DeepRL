import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

import numpy as np


class Network(nn.Module):
    """
    Semi-flexible pytorch neural net class.

    This class allows easy creation of simple (optionally convolutional)
    neural networks. Given parameters describing the basic parameters it
    automatically builds a network and manages forward passes through it.
    """

    def __init__(self, in_shape, hidden_size, out_size, conv=None, softmax=True):
        """
        Initializes and attaches (to itself) the layers of the network.

        The network constructed will consist of optional convolutional
        layers, dense layers, and an optional softmax.  The convolutional
        layers should be described by a list of tuples, each of the form
        (input channels, output channels, (square) kernel size, maxpool).
        The kernel size should be an int as non-square kernels are an
        unneeded complexity.  The maxpool should be a boolean because the
        standard (kernel size 3, stride 2) maxpool is assumed to be good
        enough.  The dense layers are described by a list of ints, where
        each int represents the number of nodes in the layer.  Softmax is
        a boolean, and if it evaluates to True then there will be a
        softmax applied to the outputs of the network before returning.
        """
        layer_shape = [a for a in in_shape]
        nn.Module.__init__(self)
        if conv is None:
            self.conv   = None
        else:
            self.conv = []
            for i, layer in enumerate(conv):
                self.conv += [nn.Conv2d(layer[0], layer[1], layer[2], padding=1)]
                self.__setattr__('c%d'%i, self.conv[-1])
                layer_shape[0] = self.conv[-1].out_channels
                if layer[3]:
                    self.conv += [nn.MaxPool2d(3, stride=2, padding=1)]
                    self.__setattr__('m%d'%i, self.conv[-1])
                    layer_shape = layer_shape[:1] + [int((a+1) / 2) for a in layer_shape[1:]]
                self.conv += [nn.ReLU()]
                self.__setattr__('cr%d'%i, self.conv[-1])

        hidden_size = [np.product(layer_shape)] + hidden_size
        self.layers = []
        for i in range(len(hidden_size) - 1):
            self.layers += [nn.Linear(hidden_size[i], hidden_size[i+1])]
            self.__setattr__('l%d'%i, self.layers[-1])
            self.layers += [nn.ReLU()]
            self.__setattr__('r%d'%i, self.layers[-1])
        self.l = nn.Linear(hidden_size[-1], out_size)

        self.softmax = softmax

    def forward(self, x):
        """Returns result of passing input x throught the network."""
        if self.conv is not None:
            for layer in self.conv:
                x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        x = self.l(x)
        if self.softmax:
            x = F.softmax(x, dim=-1)
        return x


class Agent(ABC):
    """
    Abstract agent to learn tasks sharing the OpenAI gym environment API.

    This class is a base class for building agents which learn OpenAI gym
    environments or other tasks which are accessible by an object with
    the same API as gym environments.
    """

    def __init__(self, env):
        """Store the environment for later use in training."""
        self.env    = env

    @abstractmethod
    def preprocess_state(self, state):
        pass

    @abstractmethod
    def decide(self, state):
        pass

    @abstractmethod
    def process_move(self, state, action, reward, done, next_state):
        pass

    def train(self, max_episodes, goal_return, smoothing_eps):
        """
        Trains the agent on up to max_episodes episodes.

        It will stop early if it manages to achieve an average of
        goal_return reward-per-epoch across smoothing_eps epochs.  The
        actual adjustment to the decider backing the agent should be done
        in the process_move method.
        """
        returns    = np.zeros(smoothing_eps)
        old_return = float('-inf')
        for i_ep in range(max_episodes):
            returns[i_ep % smoothing_eps] = 0
            state   = self.preprocess_state(self.env.reset())
            return_ = 0
            done    = False
            while not done:
                self.env.render()
                # Choose action
                action = None
                with torch.no_grad():
                    action = self.decide(state)

                # Take action
                next_state, reward, done, _ = self.env.step(action)

                # Process results of action
                returns[i_ep % smoothing_eps] += reward
                next_state = self.preprocess_state(next_state)
                self.process_move(state, action, reward, done, next_state)
                state = next_state


            new_return = np.mean(returns)
            if new_return >= old_return:
                print('::Network Improved: %.1f -> %.1f::' % (old_return, new_return))
                old_return = new_return

                if new_return > goal_return:
                    return True
        return False


class DeepAgent(Agent, ABC):
    """
    pytorch-based neural-net backed agentself.

    Building off of Agent, DeepAgent adds some stuff (such as learning
    rate as an instantiation parameter) that pytorch-net backed agents
    will use, thus simplifying those agents' code and reducing redundant
    code across implementations.
    """

    def __init__(self, env, alpha, gamma, use_cuda):
        """use of use_cuda allows agents to avoid managing data device."""
        Agent.__init__(self, env)
        self.alpha  = alpha
        self.gamma  = gamma
        self.Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    def preprocess_state(self, state):
        """Convert gym axis order to pytorch axis order and reduce to 1 channel."""
        axes  = list(range(len(state.shape)))
        axes  = axes[-1:] + axes[:-1]
        state = np.transpose(state, axes=axes)

        one_channel = np.mean(state, axis=0)
        return np.expand_dims(one_channel, 0)

    def decide(self, state):
        """Moves decision logic to _decision(), which takes a tensor."""
        state = self.Tensor(state).unsqueeze(0)

        action = self._decide(state)
        return action.item()

    @abstractmethod
    def _decide(self, state):
        pass

    @abstractmethod
    def update_parameters(self):
        pass


class ReplayMemory:
    """Experience replay buffer with random access."""

    def __init__(self, max_size=10000):
        self.memory   = [None] * max_size
        self.max_size = max_size
        self.i_cur    = 0

    def record(self, entry):
        self.memory[self.i_cur] = entry
        self.i_cur = (self.i_cur + 1) % self.max_size

    def recall(self, batch_size=32):
        i_max = (self.i_cur) if self.memory[-1] is None else self.max_size
        if batch_size <= i_max:
            inds = np.random.choice(i_max, size=batch_size, replace=False)
            return [self.memory[i] for i in inds]

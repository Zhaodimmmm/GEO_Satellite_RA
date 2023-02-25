# -*-coding:utf-8-*-
from builtins import print
import numpy as np
import random
import scipy.signal
from gym.spaces import Box, Discrete, Tuple, Dict
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs, rbgMap, invFlag):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, rbgMap, invFlag, act=None):
        pi = self._distribution(obs, rbgMap, invFlag)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MultiCategoricalActor(Actor):

    def __init__(self, observation, act_dim, hidden_sizes, activation):
        super().__init__()
        self.max_req = observation["Requests"][0]
        self.enb_cnt = observation["RbgMap"][0]
        self.rbg_cnt = observation["RbgMap"][1]
        self.user_per_beam = 15
        obs_dim = np.sum([np.prod(v) for k, v in observation.items()])
        self.out_dim = act_dim
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [self.out_dim], activation)

    def _distribution(self, obs, rbgMap, invFlag):
        batch_size = 1 if len(obs.shape) == 1 else obs.shape[0]
        # 根据rbgMap构造mask
        rbgMap_re = rbgMap.reshape(batch_size, self.enb_cnt, -1)
        rm1 = rbgMap.int().reshape(batch_size, -1).unsqueeze(2).expand(-1, -1, self.max_req)
        rm2 = torch.zeros((*rm1.shape[:-1], 1), dtype=torch.int, device=rm1.device)
        rmask = torch.cat((rm2, rm1), 2).bool()
        temp = invFlag.int().reshape(batch_size, self.enb_cnt, -1)
        am1 = temp.unsqueeze(2).expand(-1, -1, self.rbg_cnt, -1)
        am1 = am1.reshape(batch_size, -1, self.max_req)

        for b in range(am1.shape[0]):
            for i in range(rbgMap_re.shape[1]):
                for j in range(rbgMap_re.shape[2]):
                    if rbgMap_re[b, i, j] == 0:  # center RB
                        for k in range(am1.shape[2]):
                            if am1[b, i * rbgMap_re.shape[2] + j, k] == 0:  # center user
                                continue
                            elif am1[b, i * rbgMap_re.shape[2] + j, k] == -1:  # edge user
                                am1[b, i * rbgMap_re.shape[2] + j, k] = 1
                    elif rbgMap_re[b, i, j] == 1:  # edge RB
                        for k in range(am1.shape[2]):
                            if am1[b, i * rbgMap_re.shape[2] + j, k] == 0:  # center user
                                continue
                            elif am1[b, i * rbgMap_re.shape[2] + j, k] == -1:  # edge user
                                am1[b, i * rbgMap_re.shape[2] + j, k] = 0
        am2 = torch.zeros((*am1.shape[:-1], 1), dtype=torch.int, device=am1.device)
        amask = torch.cat((am2, am1), 2).bool()
        inp = torch.cat((obs, rbgMap.float(), invFlag.float()), 0 if len(obs.shape) == 1 else 1)
        logits = self.logits_net(inp)
        logits = logits.reshape(rmask.shape)  # 1*18*17
        logits = logits.masked_fill_(amask, -np.inf)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        if len(act.shape) == 2:  # 两个维度，第一个维度为batch_size，第二个维度为每个动作的维数
            lp = pi.log_prob(act)
            return torch.sum(lp, 1)  # 按照行为单位相加
        else:
            return torch.sum(pi.log_prob(act))


class MLPCritic(nn.Module):

    def __init__(self, observation_space, hidden_sizes, activation):
        super().__init__()
        obs_dim = np.sum([np.prod(v) for k, v in observation_space.items()])
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs.float()), -1)  # Critical to ensure v has right shape.


class RA_ActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 512, 1024, 512, 256), activation=nn.Tanh, use_cuda=True):
        super().__init__()
        action_dim = np.prod(action_space)
        self.pi = MultiCategoricalActor(observation_space, action_dim, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(observation_space, hidden_sizes, activation)
        self.use_cuda = use_cuda
        if use_cuda:
            self.pi = self.pi.cuda()
            self.v = self.v.cuda()

    def step(self, obs, rbg, flag):
        if self.use_cuda:
            obs = obs.cuda()
            rbg = rbg.cuda()
            flag = flag.cuda()

        with torch.no_grad():
            pi = self.pi._distribution(obs, rbg, flag)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            inp = torch.cat((obs, rbg.float(), flag.float()), 0 if len(obs.shape) == 1 else 1)
            v = self.v(inp)

        if self.use_cuda:
            return a.cpu().flatten().numpy(), v.cpu().numpy(), logp_a.cpu().flatten().numpy()
        else:
            return a.flatten().numpy(), v.numpy(), logp_a.flatten().numpy()

    def act(self, obs):
        return self.step(obs)[0]

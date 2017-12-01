import logging as log
log.basicConfig(filename='rewards.log', level=log.INFO)

import numpy as np
from model import CommNet
from torch.autograd import Variable
from torch import nn
import torch

BATCH_SIZE = 16
N_AGENTS = 5
N_MODELS = 10
N_LEVERS = 5
HIDSZ = 128
USE_CUDA = False

class Levers(object):
    def __init__(self):
        self.opts = {
            # model-related options
            'model': 'mlp',             # mlp | lstm | rnn, (apparently `mlp == rnn` ?)
            'hidsz': HIDSZ,             # the size of the internal state vector
            'nonlin': 'relu',           # relu | tanh | none
            'init_std': 0.2,            # STD of initial weights
            'init_hid': 0.1,            # weight of initial hidden
            # unshare_hops
            'encoder_lut': False,       # use LookupTable in encoder instead of Linear [False]
            # encoder_lut_size

            # comm-related options
            'comm_mode': 'avg',         # operation on incoming communication: avg | sum [avg]
            'comm_scale_div': 1,        # divide comm vectors by this [1]
            'comm_encoder': 1,          # encode incoming comm: 0=identity | 1=linear [0]
            'comm_decoder': 1,          # decode outgoing comm: 0=identity | 1=linear | 2=nonlin [1]
            'comm_zero_init': True,     # initialize comm weights to zero
            # comm_range
            'nactions_comm': 0,         # enable discrete communication when larger than 1 [1]
            # TODO: implement discrete comm
            # dcomm_entropy_cost
            'fully_connected': True,    # basically, all agent can talk to all agent

            # game releated
            'nmodels': N_MODELS,        # the number of models in LookupTable
            'nagents': N_AGENTS,        # the number of agents to look up
            'nactions': N_LEVERS,       # the number of agent actions

            # training
            'optim': 'rmsprop',             # optimization method: rmsprop | sgd | adam [rmsprop]
            'lrate': 1e-3,              # learning rate [0.001]
            # 'max_grad_norm':            # gradient clip value [0]
            # 'clip_grad':                # gradient clip value [0]
            # 'alpha':                    # coefficient of baseline term in the cost function [0.03]
            # 'epochs':                   # the number of training epochs [100]
            'batch_size': BATCH_SIZE,   # size of mini-batch (the number of parallel games) in each thread [16]
            # 'nworker':                  # the number of threads used for training [18]
            'reward_mult': 1,            # coeff to multiply reward for bprop [1]

            # optimizer options
            'momentum': 0,              # momentum for SGD [0]
            'wdecay': 0,                # weight decay [0]
            'rmsprop_alpha': 0.99,      # parameter of RMSProp [0.97]
            'rmsprop_eps': 1e-6,        # parameter of RMSProp [1e-06]
            'adam_beta1': 0.9,          # parameter of Adam [0.9]
            'adam_beta2': 0.999,        # parameter of Adam [0.999]
            'adam_eps': 1e-8,           # parameter of Adam [1e-08]
        }

        # init model
        self.actor = CommNet(self.opts)
        # print(self.actor)
        self._init_vars()

    def _init_vars(self):
        """initialize variables for RNN"""
        # inp is the concatenation of (h_i, c_i)
        self.inp = Variable(torch.zeros(BATCH_SIZE * N_AGENTS, HIDSZ * 2)
                    .type(torch.FloatTensor),
                    requires_grad=False)
        self.prev_hid = Variable(torch.ones(BATCH_SIZE * N_AGENTS, HIDSZ) * self.opts['init_hid'],
                            requires_grad=False)
        self.prev_cell = Variable(torch.ones(BATCH_SIZE * N_AGENTS, HIDSZ) * self.opts['init_hid'],
                            requires_grad=False)
        self.comm_in = Variable(torch.zeros(BATCH_SIZE * N_AGENTS, N_AGENTS, HIDSZ)
                        .type(torch.FloatTensor),
                        requires_grad=False)
        self.comm_mask_default = Variable(torch.ones(N_AGENTS, N_AGENTS) - torch.eye(N_AGENTS, N_AGENTS),
                                    requires_grad=False)

        # # loss function is for supervised training
        # loss_fn = torch.nn.MSELoss(size_average=False)

        # one hot for mapping action -> one hot vec
        self.emb = nn.Embedding(1, N_LEVERS)
        self.emb.weight.data = torch.eye(N_LEVERS)

        if USE_CUDA:
            self.actor = self.actor.cuda()
            self.inp = self.inp.cuda()
            self.prev_hid = self.prev_hid.cuda()
            self.prev_cell = self.prev_cell.cuda()
            self.comm_in = self.comm_in.cuda()
            self.comm_mask_default = self.comm_mask_default.cuda()

            self.emb = self.emb.cuda()
            self.emb.weight.data = torch.eye(N_LEVERS).cuda()
    
    def get_agent_ids(self):
        ids = np.array([np.random.choice(N_AGENTS, N_LEVERS, replace=False)
                        for _ in range(BATCH_SIZE)]) # ids shape: [BATCH_SIZE, N_AGENTS]
        agent_ids = Variable(torch.from_numpy(np.reshape(ids, (1, -1))),
                            requires_grad=False)
        return agent_ids

    def communicate(self, agent_ids):
        action_prob,\
        _baseline,\
        self.prev_hid,\
        self.prev_cell,\
        self.comm_in, self._action_comm = self.actor.forward(self.inp,
                                            self.prev_hid,
                                            self.prev_cell,
                                            agent_ids,
                                            self.comm_in)

        # -- determine which agent can talk to which agent?
        mask = self.comm_mask_default.view(1, N_AGENTS, N_AGENTS)
        mask = mask.expand(BATCH_SIZE, N_AGENTS, N_AGENTS)

        # if opts['fully_connected']:
        #     # -- pass all comm because it is fully connected
        # else:
        #     # -- inactive agents don't communicate
        #     # local m2 = active[t]:view(BATCH_SIZE, g_opts.nagents, 1):clone()
        #     # m2 = m2:expandAs(m):clone()
        #     # m:cmul(m2)
        #     # m:cmul(m2:transpose(2,3))
        #     # NOTE: we don't have the concept of active tensor, yet...
        #     pass

        if self.opts['comm_mode'] == 'avg':
            mask = mask / (N_AGENTS - 1)
        mask = mask / self.opts['comm_scale_div']

        # -- communication vectors for next step
        self.comm_in = self.comm_in.view(BATCH_SIZE, N_AGENTS, N_AGENTS, HIDSZ)
        # -- apply mask
        mask = mask.view(BATCH_SIZE, N_AGENTS, N_AGENTS, 1)
        mask = mask.expand_as(self.comm_in)
        self.comm_in = self.comm_in * mask
        self.comm_in = self.comm_in.transpose(1, 2)
        self.comm_in = self.comm_in.contiguous().view(BATCH_SIZE * N_AGENTS, N_AGENTS, HIDSZ)

        return action_prob, _baseline

    def get_optimizer(self):
        # optimizer
        lr = self.opts['lrate']
        optim = self.opts['optim']
        if optim == 'sgd':
            optimizer = torch.optim.SGD(self.actor.parameters(),
                                        lr=lr,
                                        momentum=self.opts['momentum'],
                                        weight_decay=self.opts['wdecay']
                                        )
        elif optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.actor.parameters(),
                                            lr=lr,
                                            alpha=self.opts['rmsprop_alpha'],
                                            eps=self.opts['rmsprop_eps'],
                                            weight_decay=self.opts['wdecay']
                                            )
        elif optim == 'adam':
            optimizer = torch.optim.Adam(self.actor.parameters(),
                                        lr=lr,
                                        betas=(self.opts['adam_beta1', self.opts['adam_beta2']]),
                                        eps=self.opts['adam_eps'],
                                        weight_decay=self.opts['wdecay']
                                        )
        else:
            raise Exception("Unsupported optimizer")
        return optimizer

    def train(self, nepisodes):
        """main training loop"""
        optimizer = self.get_optimizer()
        for i in range(nepisodes):
            optimizer.zero_grad()

            agent_ids = self.get_agent_ids()
            if USE_CUDA:
                agent_ids = agent_ids.cuda()

            # communication passes, K
            for _k in range(2):
                action_prob, _baseline = self.communicate(agent_ids)

            lever_output, loss, repeat_reward, _one_hot = self.get_reward(action_prob)
            # since multinomial creates a stochastic function, reinforce
            lever_output.reinforce(repeat_reward)

            # # supervised(broken)
            # print(action_prob)
            # batch_actions = action_prob.sum(0)
            # print("ACTION:")
            # print(batch_actions)
            # target = Variable(torch.ones(N_LEVERS) * BATCH_SIZE, requires_grad=False)
            # if USE_CUDA:
            #     target = target.cuda()
            # loss = loss_fn(batch_actions, target)

            loss.backward(retain_graph=True)
            optimizer.step()

            # reset variables for next iter
            self.prev_hid.detach_()
            self.comm_in.detach_()

            self.prev_hid.data.zero_()
            self.comm_in.data.zero_()

            if self.opts['model'] == 'lstm':
                self.prev_cell.detach_()
                self.prev_cell.data.zero_()
    
    def get_reward(self, action_prob):
        # sample an output
        lever_output = torch.multinomial(action_prob, 1)        # something like 1, 2, 3, 4, 5
        lever_ids = lever_output.view(BATCH_SIZE, N_LEVERS)     # BATCH_SIZE x N_LEVERS
        one_hot = self.emb(lever_ids)                                # BATCH_SIZE x N_LEVERS x N_LEVERS

        # sum the LEVER axis, then count the number of chosen levers
        distinct_sum = (one_hot.sum(1) > 0).sum(1).type(torch.FloatTensor)
        # reward is just the totoal count of chosen levers divided by N_LEVERS
        reward = distinct_sum / N_LEVERS * self.opts['reward_mult']
        # negate reward to optimize for loss
        loss = - reward

        printable_reward = (reward.sum(0) / BATCH_SIZE/ self.opts['reward_mult']).data[0]
        print(printable_reward)
        log.info(printable_reward)

        # repeat the rewards for each batch agent of each batch
        repeat_reward = reward \
                        .view(1, BATCH_SIZE) \
                        .data.repeat(1, N_LEVERS) \
                        .view(BATCH_SIZE * N_LEVERS, 1)
        if USE_CUDA:
            repeat_reward = repeat_reward.cuda()
        return lever_output, loss, repeat_reward, one_hot


    def save(self):
        torch.save(self.actor, "model.pt")

    def infer(self):
        self.actor = torch.load("model.pt")
        agent_ids = self.get_agent_ids()
        for _k in range(2):
            action_prob, _baseline = self.communicate(agent_ids)
        

        _, loss, _, one_hot = self.get_reward(action_prob)

        print one_hot
        print loss.sum() / BATCH_SIZE

if __name__ == "__main__":
    game = Levers()
    # game.train(1000)
    # game.save()
    game.infer()

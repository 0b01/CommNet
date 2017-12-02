"""

model.py

Model definition for CommNet

ported from (Lua)Torch

"""
import torch
from torch import nn
from linear_multi import LinearMulti

class Encoder(nn.Module):
    """
    Input -> hidden
    For other games this may be a conv + FC module

    """
    def __init__(self, in_dim, hidsz):
        super(Encoder, self).__init__()
        # self.lut = nn.Embedding(in_dim, hidsz) # in_dim agents, returns (batchsz, x, hidsz)
        # self._bias = nn.Parameter(torch.randn(hidsz), requires_grad=True)
        self.lin = nn.Linear(in_dim, hidsz)

    def forward(self, inp):
        """
        transforms env observation to hidden
        """
        return nn.ReLU()(self.lin(inp))
        # x = self.lut(inp)
        # x = torch.sum(x, 1)
        # x = x.add(self._bias)
        # return x

class CommNet(nn.Module):
    """
    Params:
        opts: options
            opts = {
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
                'comm_encoder': 0,          # encode incoming comm: 0=identity | 1=linear [0]
                'comm_decoder': 1,          # decode outgoing comm: 0=identity | 1=linear | 2=nonlin [1]
                'comm_zero_init': True,     # initialize comm weights to zero
                # comm_range
                'nactions_comm': 0,         # enable discrete communication when larger than 1 [1]
                # TODO: implement discrete comm
                # dcomm_entropy_cost
                'fully_connected': True,    # basically, all agent can talk to all agent


                'nmodels': N_MODELS,        # the number of models in LookupTable
                'nagents': N_AGENTS,        # the number of agents to look up
                'nactions': N_LEVERS,       # the number of agent actions
                'batch_size': BATCH_SIZE,   # the size of mini-batch


            }
    """
    def __init__(self, opts):
        super(CommNet, self).__init__()
        self.opts = opts

        self.nmodels = opts['nmodels']
        self.nagents = opts['nagents']
        self.model = opts['model']
        self.hidsz = opts['hidsz']
        self.nactions = opts['nactions']
        self.use_lstm = opts['model'] == 'lstm'
        self.init_std = opts['init_std']

        self.agent_ids = None # placeholder for forward

        # Comm -> hidden
        if self.opts['comm_encoder']:
            # before merging comm and hidden, use a linear layer for comm
            if self.use_lstm: # LSTM has 4x weights for gates
                self._comm2hid_linear_lstm = LinearMulti(self.nmodels, self.hidsz, self.hidsz * 4)
                if self.opts['comm_zero_init']:
                    self._comm2hid_linear_lstm.init_zero()
            else:
                self._comm2hid_linear = LinearMulti(self.nmodels, self.hidsz, self.hidsz)
                if self.opts['comm_zero_init']:
                    self._comm2hid_linear.init_zero()

        # RNN: (comm + hidden) -> hidden
        if self.use_lstm:
            self._lstm_enc = self.__build_encoder(self.hidsz * 4)
            self._lstm_linear = LinearMulti(self.nmodels, self.hidsz, self.hidsz * 4)
            self._lstm_linear.init_normal(self.init_std)
        else:
            self._rnn_enc = self.__build_encoder(self.hidsz)
            self._rnn_linear = LinearMulti(self.nmodels, self.hidsz, self.hidsz)
            self._rnn_linear.init_normal(self.init_std)

        # Action layer
        self._action_linear = LinearMulti(self.nmodels, self.hidsz, self.nactions)
        self._action_linear.init_normal(self.init_std)
        self._action_baseline_linear = LinearMulti(self.nmodels, self.hidsz, 1)
        self._action_baseline_linear.init_normal(self.init_std)

        # Comm_out
        self._comm_out_linear = LinearMulti(self.nmodels, self.hidsz, self.hidsz * self.nagents)
        self._comm_out_linear.init_zero()
        if self.opts['comm_decoder'] >= 1:
            self._comm_out_linear_alt = LinearMulti(self.nmodels, self.hidsz, self.hidsz)
            self._comm_out_linear_alt.init_zero()

        # action_comm
        nactions_comm = self.opts['nactions_comm']
        if nactions_comm > 1:
            self._action_comm_linear = LinearMulti(self.nmodels, self.hidsz, nactions_comm)

    def forward(self, inp, prev_hid, prev_cell, agent_ids, comm_in):
        """
        One communication pass. For each "hop" there may be several passes
        Params:
            inp: [batch x nagents, input_dim], s_j, state view of all agent, this
                is also used as skip connection f(h^i, c^i, h^0) where
                    h^0 = self.__rnn_enc(s^0)
            prev_hid: [batch x nagents, hidsz], previous hidden
            prev_cell: [batch x nagents, hidsz] if LSTM, otherwise None
            agent_ids: [1, batch x nagents]
            comm_in: [batch x nagents, nagents, hidsz], c^0_j = 0 for all j
        Returns:
            action_prob: q(h), can be sampled using toch.multinomial
            baseline: v(h)
            hidstate:
            cell_out:
            comm_out:
            action_comm:
        """
        self.agent_ids = agent_ids

        # c0 -> c'
        comm_ = self.__comm2hid(comm_in)

        # initalize return tuple for all of the below values    
        ret = [None] * 6

        # (c', h0) -> h1
        if self.use_lstm:
            next_hid, next_cell = self.__hid2hid(inp, comm_, prev_hid, prev_cell)
            ret[2], ret[3] = next_hid, next_cell
        else:
            next_hid = self.__hid2hid(inp, comm_, prev_hid, None)
            ret[2] = next_hid 

        action_prob, baseline = self.__action(next_hid)
        ret[0], ret[1] = action_prob, baseline

        comm_out = self.__comm_out(next_hid)
        ret[4] = comm_out

        if self.opts['nactions_comm'] > 1:
            action_comm = self.__action_comm(next_hid)
            ret[5] = action_comm

        return tuple(ret)

    def __comm2hid(self, comm_in):
        """
            c0 -> c'
        """
        # Lua Sum(2) -> Python sum(1)
        # [batch x nagents, nagents, hidsz] -> [batch x nagents, hidsz]
        comm_ = torch.sum(comm_in, 1)
        if self.opts['comm_encoder']:
            if self.use_lstm:
                comm_ = self._comm2hid_linear_lstm(comm_, self.agent_ids)
            else:
                comm_ = self._comm2hid_linear(comm_, self.agent_ids)
        return comm_

    def __hid2hid(self, inp, comm_, prev_hid, prev_cell):
        """
            (c', h0, c?) -> h1
        """
        if self.model in ('mlp', 'rnn'):
            hidstate = self._rnn(inp, comm_, prev_hid)
        elif self.use_lstm:
            hidstate, cellstate = self._lstm(inp, comm_, prev_hid, prev_cell)
            return hidstate, cellstate
        else:
            raise Exception('model not supported')
        return hidstate

    def _lstm(self, inp, comm_, prev_hid, prev_cell):
        """
            run lstm module
        """
        pre_hid = []
        pre_hid.append(self._lstm_enc(inp))                         # [batch x nagents, hidsz x 4]
        pre_hid.append(self._lstm_linear(prev_hid, self.agent_ids)) # [batch x nagents, hidsz x 4]
        pre_hid.append(comm_)                                       # [batch x nagents, hidsz]

        A = sum(pre_hid)                                            # [batch x nagents, hidsz x 4]
        B = A.view(4, self.hidsz, -1)                               # [4, hidsz, batch x nagents]
        C = torch.split(B, self.hidsz, 0)[0]

        gate_forget = nn.Sigmoid()(C[0])                            # [hidsz, batch x nagents]
        gate_write = nn.Sigmoid()(C[1])                             # [hidsz, batch x nagents]
        gate_read = nn.Sigmoid()(C[2])                              # [hidsz, batch x nagents]
        in2c = self.__nonlin()(C[3])                                # [hidsz, batch x nagents]

        cellstate = sum([
            prev_cell * gate_forget,                                # elementwise
            in2c * gate_write                                       # elementwise
        ])
        hidstate = self.__nonlin()(cellstate) * gate_read           # elementwise
        return hidstate, cellstate

    def _rnn(self, inp, comm_, prev_hid):
        """ returns RNN is just a FC layer, next hidden """
        pre_hid = []
        pre_hid.append(self._rnn_enc(inp))                          # encodes input state into feature vec
        pre_hid.append(self._rnn_linear(prev_hid, self.agent_ids))
        pre_hid.append(comm_)
        return self.__nonlin()(sum(pre_hid))

    def __action(self, hidstate):
        """
            policy and value functions share parameters
            h1 -> (pi(h1), V(h1))
        """
        action = self._action_linear(hidstate, self.agent_ids)
        action_prob = nn.Softmax()(action) # was LogSoftmax

        baseline =  self._action_baseline_linear(hidstate, self.agent_ids)

        return action_prob, baseline

    def __comm_out(self, hidstate):
        if self.opts['fully_connected']:
            comm_out = self._comm_out_linear(hidstate, self.agent_ids)
            return comm_out
        else:
            comm_out = hidstate
            if self.opts['comm_decoder'] >= 1:
                comm_out = self._comm_out_linear_alt(comm_out, self.agent_ids) # hidsz -> hidsz
                if self.opts['comm_decoder'] == 2:
                    comm_out = self.__nonlin()(comm_out)
            comm_out = comm_out.repeat(self.nagents, 1) # hidsz -> 2 x hidsz # original: comm_out = nn.Contiguous()(nn.Replicate(self.nagents, 2)(comm_out))
        return comm_out

    def __action_comm(self, hidstate):
        action_comm = self._action_comm_linear(hidstate, self.agent_ids)
        action_comm = nn.LogSoftmax()(action_comm)
        return action_comm


    def __nonlin(self):
        nonlin = self.opts['nonlin']
        if nonlin == 'tanh':
            return nn.Tanh()
        elif nonlin == 'relu':
            return nn.ReLU()
        elif nonlin == 'none':
            return Identity()
        else:
            raise Exception("wrong nonlin")

    def __build_encoder(self, hidsz):
        # in_dim = ((self.opts['visibility']*2+1) ** 2) * self.opts['nwords']
        in_dim = self.hidsz * 2
        if self.opts['encoder_lut']:                   # if there are more than 1 agent, use a LookupTable
            return Encoder(in_dim, hidsz)
        else:                                          # if only 1 agent
            return nn.Linear(in_dim, hidsz)

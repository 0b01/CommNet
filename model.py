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
        return nn.ReLU()(self.lin(inp))
        # x = self.lut(inp)
        # x = torch.sum(x, 1)
        # x = x.add(self._bias)
        # return x

class CommNet(nn.Module):
    def __init__(self, opts):
        super(CommNet, self).__init__()
        self.opts = opts
        self.nmodels = opts['nmodels']
        self.nagents = opts['nagents']
        self.hidsz = opts['hidsz']
        self.nactions = opts['nactions']
        self.use_lstm = opts['model'] == 'lstm'

        # Comm
        if self.opts['comm_encoder']:
            # before merging comm and hidden, use a linear layer for comm
            if self.use_lstm: # LSTM has 4x weights for gates
                self._comm2hid_linear = LinearMulti(self.nmodels, self.hidsz, self.hidsz * 4)
                self._comm2hid_linear.init_zero()
            else:
                self._comm2hid_linear = LinearMulti(self.nmodels, self.hidsz, self.hidsz)
                self._comm2hid_linear.init_zero()

        # RNN: (comm + hidden) -> hidden
        if self.use_lstm:
            self._rnn_enc = self.__build_encoder(self.hidsz * 4)
            self._rnn_linear = LinearMulti(self.nmodels, self.hidsz, self.hidsz * 4)
            self._rnn_linear.init_normal()
        else:
            self._rnn_enc = self.__build_encoder(self.hidsz)
            self._rnn_linear = LinearMulti(self.nmodels, self.hidsz, self.hidsz)
            self._rnn_linear.init_normal()

        # Action layer
        self._action_linear = LinearMulti(self.nmodels, self.hidsz, self.nactions)
        self._action_linear.init_normal()
        self._action_baseline_linear = LinearMulti(self.nmodels, self.hidsz, 1)
        self._action_baseline_linear.init_normal()

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
        


    def forward(self, inp, prev_hid, prev_cell, model_ids, comm_in):
        self.model_ids = model_ids
        comm2hid = self.__comm2hid(comm_in)
        # below are the return values, for next time step
        if self.use_lstm:
            hidstate, prev_cell = self.__hidstate(inp, prev_hid, prev_cell, comm2hid)
        else:
            hidstate = self.__hidstate(inp, prev_hid, prev_cell, comm2hid)

        action_prob, baseline = self.__action(hidstate)

        comm_out = self.__comm_out(hidstate)

        if self.opts['nactions_comm'] > 1:
            action_comm = self.__action_comm(hidstate)
            return (action_prob, baseline, hidstate, comm_out, action_comm)
        else:
            return (action_prob, baseline, hidstate, comm_out)

    def __comm2hid(self, comm_in):
        # Lua Sum(2) -> Python sum(1), shape: [batch x nagents, hidden]
        comm2hid = torch.sum(comm_in, 1) # XXX: sum(2) -> 0-index
        if self.opts['comm_encoder']:
            comm2hid = self._comm2hid_linear(comm2hid, self.model_ids)
        return comm2hid

    def __hidstate(self, inp, prev_hid, prev_cell, comm2hid):
        if self.opts['model'] == 'mlp' or self.opts['model'] == 'rnn':
            hidstate = self._rnn(inp, prev_hid, comm2hid)
        elif self.use_lstm:
            hidstate, cellstate = self._lstm(inp, prev_hid, prev_cell, comm2hid, self.model_ids)
            return hidstate, cellstate
        else:
            raise Exception('model not supported')
        return hidstate

    def _lstm(self, inp, prev_hid, prev_cell, comm_in):
        pre_hid = []
        pre_hid.append(self._rnn_enc(inp))
        pre_hid.append(self._rnn_linear(prev_hid, self.model_ids))
        # if comm_in:
        pre_hid.append(comm_in)
        A = sum(pre_hid)
        B = A.view(-1, 4, self.hidsz)
        C = torch.split(B, self.hidsz, 0)

        gate_forget = nn.Sigmoid()(C[0][0])
        gate_write = nn.Sigmoid()(C[0][1])
        gate_read = nn.Sigmoid()(C[0][2])
        in2c = self.__nonlin()(C[0][3])
        # print gate_forget.size(), prev_cell.size()
        # print in2c.size(), gate_write.transpose(0,1).size()
        cellstate = sum([
            torch.matmul(gate_forget, prev_cell),
            torch.matmul(in2c.transpose(0,1), gate_write)
        ])
        hidstate = torch.matmul(self.__nonlin()(cellstate), gate_read)
        return hidstate, cellstate

    def _rnn(self, inp, prev_hid, comm_in):
        pre_hid = []
        pre_hid.append(self._rnn_enc(inp))

        pre_hid.append(self._rnn_linear(prev_hid, self.model_ids))
        # if comm_in:
        pre_hid.append(comm_in)

        sum_pre_hid = sum(pre_hid)
        hidstate = self.__nonlin()(sum_pre_hid)
        return hidstate

    def __action(self, hidstate):
        action = self._action_linear(hidstate, self.model_ids)
        action_prob = nn.Softmax()(action) # was LogSoftmax

        baseline =  self._action_baseline_linear(hidstate, self.model_ids)

        return action_prob, baseline

    def __comm_out(self, hidstate):
        if self.opts['fully_connected']:
            comm_out = self._comm_out_linear(hidstate, self.model_ids)
            amount = self.opts['nagents'] - 1
            return comm_out / amount
        else:
            comm_out = hidstate
            if self.opts['comm_decoder'] >= 1:
                comm_out = self._comm_out_linear_alt(comm_out, self.model_ids) # hidsz -> hidsz
                if self.opts['comm_decoder'] == 2:
                    comm_out = self.__nonlin()(comm_out)
            comm_out.repeat(self.nagents, 2) # hidsz -> 2 x hidsz # original: comm_out = nn.Contiguous()(nn.Replicate(self.nagents, 2)(comm_out))
        return comm_out

    def __action_comm(self, hidstate):
        action_comm = self._action_comm_linear(hidstate, self.model_ids)
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


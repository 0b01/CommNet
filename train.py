# import logging as log
# # set logger
import numpy as np
from model import CommNet
from torch.autograd import Variable
from torch import nn
import torch

BATCH_SIZE = 2
N_AGENTS = 5
N_MODELS = 10
N_LEVERS = 5
HIDSZ = 128
USE_CUDA = False


def train(nepisodes):
    """train for number of nepisodes"""
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

    actor = CommNet(opts)
    print(actor)


    # inp is the concatenation of (h_i, c_i)
    inp = Variable(torch.zeros(BATCH_SIZE * N_AGENTS, HIDSZ * 2)
                        .type(torch.FloatTensor),
                   requires_grad=False)
    prev_hid = Variable(torch.ones(BATCH_SIZE * N_AGENTS, HIDSZ) * opts['init_hid'],
                        requires_grad=False)
    prev_cell = Variable(torch.ones(BATCH_SIZE * N_AGENTS, HIDSZ) * opts['init_hid'], 
                         requires_grad=False)
    comm_in = Variable(torch.zeros(BATCH_SIZE * N_AGENTS, N_AGENTS, HIDSZ)
                            .type(torch.FloatTensor),
                       requires_grad=False)



    learning_rate = 1e-2
    optimizer = torch.optim.SGD(actor.parameters(), lr=learning_rate, momentum=0.9)
    loss_fn = torch.nn.MSELoss(size_average=False)

    # one hot for mapping action -> one hot vec
    emb = nn.Embedding(1, N_LEVERS)
    emb.weight.data = torch.eye(N_LEVERS)

    if USE_CUDA:
        actor = actor.cuda()
        inp = inp.cuda()
        prev_hid = prev_hid.cuda()
        prev_cell = prev_cell.cuda()
        comm_in = comm_in.cuda()
        emb = emb.cuda()
        emb.weight.data = torch.eye(N_LEVERS).cuda()

    # main training loop
    for i in range(nepisodes):
        ids = np.array([np.random.choice(N_AGENTS, N_LEVERS, replace=False)
                        for _ in range(BATCH_SIZE)]) # ids shape: [BATCH_SIZE, N_AGENTS]
        model_ids = Variable(torch.from_numpy(np.reshape(ids, (1, -1))),
                            requires_grad=False)

        if USE_CUDA:
            model_ids = model_ids.cuda()

        # print('iter: ', i, '------------' * 5)
        optimizer.zero_grad()

        for _k in range(2):
            action_prob,\
            _baseline,\
            prev_hid,\
            prev_cell,\
            comm_in, _action_comm = actor.forward(inp,
                                                  prev_hid,
                                                  prev_cell,
                                                  model_ids,
                                                  comm_in)

            # # TODO: figure out how to do comm_mode: 'avg'

            # if g_opts.comm or g_opts.nactions_comm > 1 then
            #     -- determine which agent can talk to which agent?
            #     local m = comm_mask_default:view(1, g_opts.nagents, g_opts.nagents)
            #     m = m:expand(g_opts.batch_size, g_opts.nagents, g_opts.nagents):clone()

            #     if g_opts.fully_connected then
            #         -- pass all comm because it is fully connected
            #     else
            #         -- inactive agents don't communicate
            #         local m2 = active[t]:view(g_opts.batch_size, g_opts.nagents, 1):clone()
            #         m2 = m2:expandAs(m):clone()
            #         m:cmul(m2)
            #         m:cmul(m2:transpose(2,3))
            #     end

            #     if g_opts.comm_range > 0 then
            #         -- far away agents can't communicate
            #         for i, g in pairs(batch) do
            #             for s = 1, g_opts.nagents do
            #                 for d = 1, g_opts.nagents do
            #                     local dy = math.abs(get_agent(g, s).loc.y - get_agent(g, d).loc.y)
            #                     local dx = math.abs(get_agent(g, s).loc.x - get_agent(g, d).loc.x)
            #                     local r = math.max(dy, dx)
            #                     if r > g_opts.comm_range then
            #                         m[i][s][d] = 0
            #                     end
            #                 end
            #             end
            #         end
            #     end

            #     if g_opts.comm_mode == 'avg' then
            #         -- average comms by dividing by number of agents
            #         m:cdiv(m:sum(2):expandAs(m):clone():add(m:eq(0):float()))
            #     end
            #     m:div(g_opts.comm_scale_div)
            #     comm_mask[t] = m
            # end

            # if g_opts.comm then
            #     -- communication vectors for next step
            #     local h = out[g_model_outputs['comm_out']]:clone()
            #     h = h:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, g_opts.hidsz)
            #     -- apply mask
            #     local m = comm_mask[t]
            #     m = m:view(g_opts.batch_size, g_opts.nagents, g_opts.nagents, 1)
            #     m = m:expandAs(h):clone()
            #     h:cmul(m)
            #     comm_state = h:transpose(2,3):clone()
            #     comm_state:resize(g_opts.batch_size * g_opts.nagents, g_opts.nagents, g_opts.hidsz)
            # end
            comm_in = comm_in.view(BATCH_SIZE, N_AGENTS, N_AGENTS, HIDSZ)
            comm_in = comm_in.transpose(1, 2)
            comm_in = comm_in.contiguous().view(BATCH_SIZE * N_AGENTS, N_AGENTS, HIDSZ)

        lever_output = torch.multinomial(action_prob, 1)
        lever_ids = lever_output.view(BATCH_SIZE, N_LEVERS)
        one_hot = emb(lever_ids) # 1x5x5
        distinct_sum = (one_hot.sum(1) > 0).sum(1).type(torch.FloatTensor)
        reward = distinct_sum / N_LEVERS
        loss = -reward
        print(reward.sum(0) / BATCH_SIZE)
        repeat_reward = reward.view(1, BATCH_SIZE).data.repeat(1, N_LEVERS).view(BATCH_SIZE * N_LEVERS, 1)
        if USE_CUDA:
            repeat_reward = repeat_reward.cuda()
        lever_output.reinforce(repeat_reward)

        # # supervised
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
        prev_hid.detach_()
        comm_in.detach_()
        action_prob.detach_()

        prev_hid.data.zero_()
        comm_in.data.zero_()
        action_prob.data.zero_()

        if opts['model'] == 'lstm':
            prev_cell.detach_()
            prev_cell.data.zero_()


if __name__ == "__main__":
    train(7810)

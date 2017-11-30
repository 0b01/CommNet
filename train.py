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


def train(episode):
    opts = {
        'comm_encoder': False,      # bool
        'nonlin': 'relu',           # relu | tanh | none
        'nactions_comm': 0,         # discrete communication through action
        'nwords': 1,                # TODO: figure out what this does
        'encoder_lut_nil': None,
        'encoder_lut': True,
        'hidsz': HIDSZ,
        'nmodels': N_MODELS,
        'nagents': N_AGENTS,
        'nactions': N_LEVERS,
        'model': 'mlp',
        'batch_size': BATCH_SIZE,
        'fully_connected': True,
        'comm_decoder': 0,
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
    for i in range(episode):
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

            # TODO: figure out how to do comm_mode: 'avg'
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

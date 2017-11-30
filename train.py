# import logging as log
# # set logger
import numpy as np
from model import CommNet
from torch.autograd import Variable
from torch import nn
import torch

N_AGENTS = 5
BATCH_SIZE = 1
LEVER = 5
HIDSZ = 10


def train(episode):
    opts = {
        'comm_encoder': False,
        'nonlin': 'relu',
        'nactions_comm': 0,
        'nwords': 1,
        'encoder_lut_nil': None,
        'encoder_lut': True,
        'hidsz': HIDSZ,
        'nmodels': N_AGENTS,
        'nagents': N_AGENTS,
        'nactions': LEVER,
        'model': 'mlp',
        'batch_size': BATCH_SIZE,
        'fully_connected': True,
        'comm_decoder': 0,
    }

    actor = CommNet(opts).cuda()
    print(actor)


    inp = Variable(torch.zeros(BATCH_SIZE * N_AGENTS, 1).type(torch.LongTensor), requires_grad=False) # input is none
    prev_hid = Variable(torch.zeros(BATCH_SIZE * N_AGENTS, HIDSZ)
                             .type(torch.FloatTensor), requires_grad=False)
    prev_cell = Variable(torch.zeros(BATCH_SIZE * N_AGENTS, HIDSZ), requires_grad=False)

    comm_in = Variable(
        torch.zeros(BATCH_SIZE * N_AGENTS,
                   N_AGENTS,
                   HIDSZ)
             .type(torch.FloatTensor), requires_grad=False)


    learning_rate = 1e-3
    optimizer = torch.optim.SGD(actor.parameters(), lr=learning_rate, momentum=0.9)
    loss_fn = torch.nn.MSELoss(size_average=False)

    # one hot for mapping action
    emb = nn.Embedding(1, 5).cuda() 
    emb.weight.data = torch.eye(5).cuda()

    # clip = 1e-1
    # torch.nn.utils.clip_grad_norm(actor.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(actor._action_baseline_linear.parameters(), clip)
    # # torch.nn.utils.clip_grad_norm(actor._action_comm_linear.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(actor._action_linear.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(actor._comm_out_linear.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(actor._comm2hid_linear.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(actor._comm_out_linear_alt.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(actor._rnn_enc.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(actor._rnn_linear.parameters(), clip)
    # torch.nn.utils.clip_grad_norm(actor._action_baseline_linear.parameters(), clip)
    ids = np.array([np.random.choice(N_AGENTS, LEVER, replace=False)
                    for _ in range(BATCH_SIZE)])
    # ids shape: [BATCH_SIZE, 5]
    model_ids = Variable(torch.from_numpy(np.reshape(ids, (1, -1))), requires_grad=False)

    for i in range(episode):
        print(i, '------------' * 5)
        # print([ w.data[0] for w in list(actor.parameters()) ])
        optimizer.zero_grad()

        action_prob, _baseline, prev_hid, comm_in = actor.forward(inp.cuda(),
                                                                  prev_hid.cuda(),
                                                                  prev_cell.cuda(),
                                                                  model_ids.cuda(),
                                                                  comm_in.cuda())
        comm_in = comm_in.contiguous().view(BATCH_SIZE * N_AGENTS, N_AGENTS, HIDSZ)
        action_prob, _baseline, prev_hid, comm_in = actor.forward(inp.cuda(),
                                                                  prev_hid.cuda(),
                                                                  prev_cell.cuda(),
                                                                  model_ids.cuda(),
                                                                  comm_in.cuda())
        comm_in = comm_in.contiguous().view(BATCH_SIZE * N_AGENTS, N_AGENTS, HIDSZ)
        action_prob, _baseline, prev_hid, comm_in = actor.forward(inp.cuda(),
                                                                  prev_hid.cuda(),
                                                                  prev_cell.cuda(),
                                                                  model_ids.cuda(),
                                                                  comm_in.cuda())

        # comm_in = comm_in.view(BATCH_SIZE, N_AGENTS, N_AGENTS, HIDSZ)
        # comm_in = comm_in.transpose(1, 2)
        comm_in = comm_in.contiguous().view(BATCH_SIZE * N_AGENTS, N_AGENTS, HIDSZ)

        # lever_output = torch.multinomial(action_prob, 1)
        # lever_ids = lever_output.view(BATCH_SIZE, LEVER)
        # one_hot = emb(lever_ids) # 1x5x5
        # distinct_sum = (one_hot.sum(1) > 0).sum(1).type(torch.FloatTensor)
        # reward = distinct_sum / LEVER
        # loss = - reward
        # print(reward.sum(0) / BATCH_SIZE)
        # repeat_reward = reward.view(1, BATCH_SIZE).data.repeat(1, LEVER).view(BATCH_SIZE * LEVER, 1)
        # lever_output.reinforce(repeat_reward.cuda())

        print(action_prob)
        batch_actions = action_prob.sum(0)
        print("ACTION:")
        print(batch_actions)
        target = Variable(torch.ones(LEVER) * BATCH_SIZE, requires_grad=False).cuda()
        loss = loss_fn(batch_actions, target)

        loss.backward(retain_graph=True)
        optimizer.step()

if __name__ == "__main__":
    train(700)

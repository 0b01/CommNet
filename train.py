# import logging as log
# # set logger
import numpy as np
from model import CommNet
from torch.autograd import Variable
from torch import nn
import torch

N_AGENTS = 5
BATCH_SIZE = 16
LEVER = 5
HIDSZ = 128


def train(episode):
    opts = {
        'comm_encoder': False,
        'nonlin': 'relu',
        'nactions_comm': 0,
        'nwords': 1,
        'encoder_lut_nil': None,
        'encoder_lut': True,
        'hidsz': HIDSZ,
        'nmodels': 20,
        'nagents': N_AGENTS,
        'nactions': LEVER,
        'model': 'mlp',
        'batch_size': BATCH_SIZE,
        'fully_connected': True,
        'comm_decoder': 0,
    }

    actor = CommNet(opts).cuda()
    # print(actor)


    # inp is the concatenation of (h_i, c_i)
    inp = Variable(torch.zeros(BATCH_SIZE * N_AGENTS, HIDSZ * 2)
                        .type(torch.FloatTensor),
                   requires_grad=False).cuda()
    prev_hid = Variable(torch.zeros(BATCH_SIZE * N_AGENTS, HIDSZ)
                             .type(torch.FloatTensor),
                        requires_grad=False).cuda()
    prev_cell = Variable(torch.zeros(BATCH_SIZE * N_AGENTS, HIDSZ), 
                         requires_grad=False).cuda()

    comm_in = Variable(
        torch.zeros(BATCH_SIZE * N_AGENTS,
                   N_AGENTS,
                   HIDSZ)
             .type(torch.FloatTensor), requires_grad=False).cuda()


    learning_rate = 1e-2
    optimizer = torch.optim.SGD(actor.parameters(), lr=learning_rate, momentum=0)
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


    for i in range(episode):
        ids = np.array([np.random.choice(N_AGENTS, LEVER, replace=False)
                        for _ in range(BATCH_SIZE)])
        # ids shape: [BATCH_SIZE, 5]
        model_ids = Variable(torch.from_numpy(np.reshape(ids, (1, -1))),
                            requires_grad=False).cuda()

        # print(i, '------------' * 5)
        optimizer.zero_grad()

        for _k in range(2):
            action_prob, _baseline, prev_hid, comm_in = actor.forward(inp,
                                                                    prev_hid,
                                                                    prev_cell,
                                                                    model_ids,
                                                                    comm_in)

            comm_in = comm_in.view(BATCH_SIZE, N_AGENTS, N_AGENTS, HIDSZ)
            comm_in = comm_in.transpose(1, 2)
            comm_in = comm_in.contiguous().view(BATCH_SIZE * N_AGENTS, N_AGENTS, HIDSZ)

        lever_output = torch.multinomial(action_prob, 1)
        lever_ids = lever_output.view(BATCH_SIZE, LEVER)
        one_hot = emb(lever_ids) # 1x5x5
        distinct_sum = (one_hot.sum(1) > 0).sum(1).type(torch.FloatTensor)
        reward = distinct_sum / LEVER
        loss = -reward
        print(reward.sum(0).data[0] / BATCH_SIZE)
        repeat_reward = reward.view(1, BATCH_SIZE).data.repeat(1, LEVER).view(BATCH_SIZE * LEVER, 1)
        lever_output.reinforce(repeat_reward.cuda())

        # # supervised
        # print(action_prob)
        # batch_actions = action_prob.sum(0)
        # print("ACTION:")
        # print(batch_actions)
        # target = Variable(torch.ones(LEVER) * BATCH_SIZE, requires_grad=False).cuda()
        # loss = loss_fn(batch_actions, target)

        loss.backward(retain_graph=True)
        optimizer.step()

        # reset variables for next iter
        prev_hid.detach_()
        prev_cell.detach_()
        comm_in.detach_()
        action_prob.detach_()

        prev_hid.data.zero_()
        prev_cell.data.zero_()
        comm_in.data.zero_()
        action_prob.data.zero_()


if __name__ == "__main__":
    train(7810)

# Communication Neural Network (CommNet)

[[Paper pdf]](https://arxiv.org/pdf/1605.07736.pdf)

[[original Torch impl]](https://github.com/facebookresearch/CommNet/)


Ported to PyTorch from Torch. This network enables neural network based agents to communicate for cooperation.


# Training

To train the network

```
python train.py
```

# Levers Task

Each agent must pull a different lever after 2 communication passes. Since the agents have to cooperate, levers game is a sanity check for the implementation.

![accuracy](https://raw.githubusercontent.com/rickyhan/CommNet/master/accuracy.png)

# Options

```python
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
```

# TODO

- [x] Implement LSTM module

- [x] `'comm_mode': 'avg'` is broken

- [ ] Implement discrete communication through action

- [x] Hyperparameter tuning
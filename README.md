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

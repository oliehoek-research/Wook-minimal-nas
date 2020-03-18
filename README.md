*Note: short report on experiment with changes can be found in [report.md](report.md).*

---

# Minimal implementation of a Neural Architecture Search system


This repository implements a simple Neural Architecture Search (NAS) system in PyTorch. Heavily inspired by the work of [Barret Zoph & Quoc V. Le (2016)](https://arxiv.org/abs/1611.01578).

## How to use

You can run the experiment by calling ```train.py``` directly:

```
python train.py
```

You will find a number of hyper-parameters in the files that you can alter directly. This is to prevent unnecessary clutter in this minimal implementation.

## Dataset

As proof of concept, the [HalfMoon 2D dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) generated by scikit-learn is used. I generate 1000 samples with a noise level of 0.2, which ensures fast training of child networks while still requiring some non-linearity to accurately classify samples into one of the two classes. The figure below shows a few samples from the dataset.

![halfmoon](https://i.imgur.com/ynSnpMU.png)

## State space

A small action space has been used. It contains the following freely mixed components: linear layers with 1, 2, 4, 8 or 16 neurons as well as the ```Sigmoid```, ```Tanh```, ```ReLU``` and ```LeakyReLU``` non-linear activation functions. Any generated network will have an output layer with 2 outputs for classification. The controller has been restricted to generate child networks with at most 6 hidden layers.

## Controller

The controller is a single [Gated Recurrent Unit](https://pytorch.org/docs/stable/nn.html#gru) (GRU) cell followed by a linear layer to produce a distribution over actions that we can sample from. The hidden state from the recurrent cell is used to encode the current state. The controller is trained with the [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) policy gradient method with an added exponentially decaying entropy term to encourage exploration. Accuracy of the child network is used as reward signal during training.

## Results

Accuracy of child networks generated by the controller over the course of 5000 rollouts is displayed in the figure below. An accuracy of 91% is achieved at test time.

![results](https://i.imgur.com/ADAPV0g.png)

An example of a generated child network is:

```[8, Tanh, 8, ReLU, 4, ReLU, 2]```

where ```2``` is the output layer.

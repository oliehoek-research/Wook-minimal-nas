import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from model import Net


class Controller(nn.Module):

    def __init__(self):
        super(Controller, self).__init__()

        self.hidden_size = 32
        self.num_layers = 1
        self.neurons= [1, 2, 4, 8, 16]
        self.activations = ['Sigmoid', 'Tanh', 'ReLU', 'LeakyReLU']
        self.params_list = [self.neurons, self.activations]
        self.params = [len(self.neurons), len(self.activations)]

        self.embed = nn.Embedding(
            num_embeddings=sum(self.params),
            embedding_dim=self.hidden_size
        )
        self.cell = nn.GRUCell(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size
        )
        self.decoders = nn.ModuleList(
            [nn.Linear(
                in_features=self.hidden_size,
                out_features=action_size
            ) for action_size in self.params]
        )

        self.epsilon = 0.8
        self.gamma = 1.0
        self.beta = 0.0001  ## 0.01
        self.clip_norm = 0
        self.log_probs = []
        self.actions = []
        self.entropies = []
        self.reward = None

        self.baseline = None
        self.decay = 0.95
        self.adv = 0

        self.minibatch_size = 20

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, x, h):
        for layer_idx in range(self.num_layers):
            for param_idx in range(len(self.params)):
                emb = self.embed(x).squeeze(1) if layer_idx + param_idx != 0 else x  ## x (B, H)
                h = self.cell(emb, h)  ## (B, H)
                logits = self.decoders[param_idx](h)  ## (B, F)

                action_idx_v = torch.distributions.Categorical(logits=logits).sample().detach()
                action_idx = action_idx_v.item()
                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs)

                ## assume batch_size = 1
                action = self.params_list[param_idx][action_idx]
                self.actions.append(action)
                entropy = -(log_probs * probs).sum(dim=-1).item()
                self.entropies.append(entropy)
                log_prob = log_probs[0, action_idx].item()
                self.log_probs.append(log_prob)

                x = action_idx_v + sum(self.params[:param_idx]) if param_idx > 0 else action_idx_v

    def generate_rollout(self, iter_train, iter_dev, verbose=False):
        self.log_probs = []
        self.actions = []
        self.entropies = []
        self.reward = None
        self.adv = 0

        state = torch.zeros(1, self.hidden_size)
        self.reward = 0
        self(state, None)

        if verbose:
            print('Generated network:', self.actions)

        net = Net(self.actions)
        accuracy = net.fit(iter_train, iter_dev)
        self.reward += accuracy

        # moving average baseline
        if self.baseline is None:
            self.baseline = self.reward
        else:
            self.baseline = self.decay * self.baseline + (1 - self.decay) * self.reward
        self.adv = self.reward - self.baseline

        return self.reward, self.adv

    def optimize(self, idx):
        G = torch.ones(1) * self.adv  ## self.reward
        loss = 0

        for i in reversed(range(len(self.log_probs))):
            G = self.gamma * G
            loss = loss - (self.log_probs[i]*Variable(G)) - self.beta * self.entropies[i]

        loss /= len(self.log_probs)
        loss /= self.minibatch_size

        if idx == 0:
            self.optimizer.zero_grad()

        loss = Variable(loss, requires_grad = True)
        loss.backward(retain_graph=True)

        if self.clip_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)

        if idx > 0 and idx % self.minibatch_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return float(loss.data.numpy())

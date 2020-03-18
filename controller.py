import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from model import Net

class Controller(nn.Module):

    def __init__(self, num_actions=10, hidden_size=64):
        super(Controller, self).__init__()

        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.epsilon = 0.8
        self.gamma = 1.0
        self.beta = 0.0001
        self.max_depth = 6
        self.clip_norm = 0
        self.log_probs = []
        self.actions = []
        self.entropies = []
        self.reward = None

        self.baseline = None
        self.decay = 0.95
        self.adv = 0

        self.index_to_action = {
            0: 1,
            1: 2,
            2: 4,
            3: 8,
            4: 16,
            5: 'Sigmoid',
            6: 'Tanh',
            7: 'ReLU',
            8: 'LeakyReLU',
            9: 'EOS'
        }

        self.embed = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=self.hidden_size
        )

        self.cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=num_actions
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)


    def forward(self, x, h, is_embed):
        '''
        paramters: x (H,) or (L,)
        paramters: h (B, H)
        '''
        if not is_embed:
            embed = self.embed(x)  ## (B, H)
        else:
            embed = x.unsqueeze(dim=0)  ## (B, H)
        ## h = h.unsqueeze(dim=0)  ## (B, H)

        h = self.cell(embed, h)  ## (B, H)
        x = self.fc(h)  ## (B, L)

        ## h = h.squeeze(dim=0)  ## (H,)
        x = x.squeeze(dim=0)  ## (L,)

        return x, h


    def generate_rollout(self, iter_train, iter_dev, verbose=False):
        self.log_probs = []
        self.actions = []
        self.entropies = []
        self.reward = None
        self.adv = 0

        state = None
        input = torch.zeros(self.hidden_size)
        is_embed = True
        terminate = False
        self.reward = 0

        while not terminate:
            logits, state = self(input, state, is_embed)

            idx = torch.distributions.Categorical(logits=logits).sample().detach()
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs)
            self.log_probs.append(log_probs[idx])

            action = self.index_to_action[int(idx)]
            self.actions.append(action)

            entropy = -(log_probs * probs).sum(dim=-1)
            self.entropies.append(entropy)

            terminate = (action == 'EOS') or (len(self.actions) == self.max_depth)
            is_embed = False
            input = Variable(torch.LongTensor([idx]), requires_grad=False)

        if verbose:
            print('\nGenerated network:')
            print(self.actions)

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


    def optimize(self):
        G = torch.ones(1) * self.adv  ## self.reward
        loss = 0

        for i in reversed(range(len(self.log_probs))):
            G = self.gamma * G
            loss = loss - (self.log_probs[i]*Variable(G)) - self.beta * self.entropies[i]

        loss /= len(self.log_probs)

        self.optimizer.zero_grad()
        loss.backward()

        if self.clip_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)

        self.optimizer.step()

        return float(loss.data.numpy())

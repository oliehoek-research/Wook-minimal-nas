import numpy as np
import sklearn.datasets
import torch
from controller import Controller
from torch.utils import data
import matplotlib.pyplot as plt
import pickle as pkl

class Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(p_val=0.1, p_test=0.2):
    np.random.seed(0)
    num_samples = 1000
    ## X, y = sklearn.datasets.make_moons(num_samples, noise=0.2)
    X, y = sklearn.datasets.make_circles(num_samples, noise=0.05)

    train_end = int(len(X)*(1-p_val-p_test))
    val_end = int(len(X)*(1-p_test))

    ds_train = Dataset(X[:train_end], y[:train_end])
    ds_dev = Dataset(X[train_end:val_end], y[train_end:val_end])
    ds_test = Dataset(X[val_end:], y[val_end:])

    dl_train = data.DataLoader(ds_train, batch_size=32, shuffle=True)
    dl_dev = data.DataLoader(ds_dev, batch_size=32, shuffle=True)
    dl_test = data.DataLoader(ds_test, batch_size=32, shuffle=True)

    return dl_train, dl_dev, dl_test


if __name__ == '__main__':
    dl_train, dl_dev, dl_test = load_dataset()
    controller = Controller()
    num_rollouts = 2000

    rewards = []
    losses = []
    advs = []

    print('Training controller...')

    for i in range(num_rollouts):
        reward, adv = controller.generate_rollout(dl_train, dl_dev)
        loss = controller.optimize()
        ## controller.beta *= 0.99

        rewards.append(reward)
        advs.append(adv)
        losses.append(loss)

        if (i + 1) % 100 == 0 and i > 0:
            ## print(f'Rollout {i}, mean reward: {np.mean(rewards[-100:])}, beta: {controller.beta}, loss: {np.mean(losses[-100:])}')
            print(f'Rollout {i + 1}, mean reward: {np.mean(rewards[-100:])}, advantage: {advs[-1]}, beta: {controller.beta}, loss: {np.mean(losses[-100:])}, network: {controller.actions}')

    with open('rewards_losses.pkl', 'wb') as handle:
        pkl.dump((rewards, advs, losses), handle, protocol=pkl.HIGHEST_PROTOCOL)

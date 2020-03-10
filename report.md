## Results of original work
It looks like the source code not working properly. For example, the following shows test results with 4 different runs over 5,000 rollouts without any changes.  

| # | 1 | 2 | 3 | 4 |
| -- | --- | --- | --- | --- |
|Accuracy | 0.81 | 0.43 | 0.58 | 0.70 |
|Generated actions for child network | ```[1, 'Sigmoid', 'EOS']``` | ```['ReLU', 16, 16, 16, 16, 16]``` | ```[16, 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU']``` | ```['Sigmoid', 'Sigmoid', 'EOS']``` |
|Rollout plot | ![test_1](https://user-images.githubusercontent.com/59391289/76166792-876cd500-6161-11ea-9f26-36e230f966d6.png) | ![test_2](https://user-images.githubusercontent.com/59391289/76166806-b2efbf80-6161-11ea-8cad-89fb80b4f29c.png) | ![test_3](https://user-images.githubusercontent.com/59391289/76166865-49bc7c00-6162-11ea-8edb-08eeafbab350.png) | ![test_4](https://user-images.githubusercontent.com/59391289/76166889-74a6d000-6162-11ea-9c0f-ce886169b36c.png) |

## Modification
Although implemented in a minimal fashion, several important points different from the original work of [Barret Zoph & Quoc V. Le (2016)](https://arxiv.org/abs/1611.01578) are found and changed accordingly to see if they are helpful.
- Zero input to all controller cells :arrow_right: Autoregressive connection with ```embedding```layers added
- Arbitrary combination of actions allowed out of [1, 2, 4, 8, 16, ```Sigmoid```, ```Tanh```, ```ReLU```, ```LeakyReLU```, ```EOS```], at most 6 actions to select with output layer of 2 neurons, some negative rewards given to unusual configurations :arrow_right: The order is fixed. ```Linear``` layers should be followed by activation function per each, plus the output layer.  

The following features are also added to improve the agent training
- :arrow_right: Moving average baseline ([ref](https://github.com/carpedm20/ENAS-pytorch/blob/master/trainer.py))
- :arrow_right: Weight update using averaged minibatch of gradient estimates in a serial way ([ref](https://github.com/TDeVries/enas_pytorch/blob/master/train.py))

<kbd>
  <img src="https://user-images.githubusercontent.com/59391289/76312175-bf475a00-62d2-11ea-903f-b79a0a63cd77.png" width="500">
  <img src="https://user-images.githubusercontent.com/59391289/76312620-b905ad80-62d3-11ea-94bc-369fda11a377.png" width="500">
</kbd>

## Results of revised work
Since the dataset used is quite simple for classification, a network with one hidden layer is used for test. The output shows some variance, which needs to play with hyperparameters. More complex dataset should be better for verification.
<kbd>
  <img src="https://user-images.githubusercontent.com/59391289/76348189-1287cf80-6308-11ea-9c17-674e126f399a.png" width="500">
  <img src="https://user-images.githubusercontent.com/59391289/76348345-54b11100-6308-11ea-98a0-870c363cb439.png" width="500">
</kbd>

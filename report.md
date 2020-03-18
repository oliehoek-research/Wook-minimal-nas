## Original work
The results may not be consistent. For example, the following shows 3 different runs without modification.  

| # | 1 | 2 | 3 |
| :-: | :-: | :-: | :-: |
| Accuracy | 0.81 | 0.43 | 0.58 |
| Generated child network | ```[1, 'Sigmoid', 2]``` | ```['ReLU', 16, 16, 16, 16, 16, 2]``` | ```[16, 'ReLU', 'ReLU', 'ReLU', 'ReLU', 'ReLU', 2]``` |
| Rollout plot | ![test_1](https://user-images.githubusercontent.com/59391289/76166792-876cd500-6161-11ea-9f26-36e230f966d6.png) | ![test_2](https://user-images.githubusercontent.com/59391289/76166806-b2efbf80-6161-11ea-8cad-89fb80b4f29c.png) | ![test_3](https://user-images.githubusercontent.com/59391289/76166865-49bc7c00-6162-11ea-8edb-08eeafbab350.png) |

## Test with changes
Although intended for minimal implementation, several points different from the paper of [Barret Zoph & Quoc V. Le (2016)](https://arxiv.org/abs/1611.01578) are found and changed accordingly to see if they are helpful.

##### Controller
- Cells connected in autoregressive fashion, with ```embedding```layers added in between.

##### Training
- Moving average baseline to reduce variance
- Some change in hyperparameters, reward and others

##### Dataset
- Tested with [2D full-moon](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html) as well as [half-moon](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)   

<kbd>
  <img src="https://user-images.githubusercontent.com/59391289/76312175-bf475a00-62d2-11ea-903f-b79a0a63cd77.png" width="400">
  <img src="https://user-images.githubusercontent.com/59391289/77008206-00400e00-6966-11ea-862b-1e067252f27d.png" width="400">
</kbd>

## Experiment result
Application of the **baseline** turns out to be a key factor for convergence.

| Dataset | Half-moon | Full-moon |
| :-: | :-: | :-: |
| Reward | ![reward1](https://user-images.githubusercontent.com/59391289/77013945-7b5af180-6971-11ea-87cc-81449606ca4c.png) | ![reward2](https://user-images.githubusercontent.com/59391289/77014148-ed333b00-6971-11ea-8e59-fce80aef2be1.png) |
| Advantage | ![adv1](https://user-images.githubusercontent.com/59391289/77014051-b3623480-6971-11ea-82e1-0cd95d993e9e.png) | ![adv2](https://user-images.githubusercontent.com/59391289/77014189-04722880-6972-11ea-8a44-5917dca8e64a.png) |

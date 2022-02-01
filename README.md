# PPO-Pytorch
A clean and minimal implementation of PPO (Proximal Policy Optimization) algorithm in Pytorch, for continuous action spaces.

## References
* "Proximal Policy Optimization Algorithms", Schulman et al. [Link](https://arxiv.org/abs/1707.06347).

## Tested on

* Cartpole Swingup ([Deepmind Control Suite](https://github.com/deepmind/dm_control/tree/master/dm_control/suite)) - Swing up and balance an unactuated pole by applying forces to a cart at its base.

<p align="center">
<img src=".media/ppo_cartpole_swingup.png" width="50%" height="50%"/>
</p>

<p align="center">
<img src=".media/ppo_cartpole_swingup.gif" width="50%" height="50%"/>
</p>

* Reacher Hard ([Deepmind Control Suite](https://github.com/deepmind/dm_control/tree/master/dm_control/suite)) - Control a two-link robotic arm to reach a randomized target location.

<p align="center">
<img src=".media/ppo_reacher_hard.png" width="50%" height="50%"/>
</p>

<p align="center">
<img src=".media/ppo_reacher_hard.gif" width="50%" height="50%"/>
</p>

## A few implementation details 
When implementing PPO, a few tricks are necessary for good performance across environments. These are not clearly mentioned in the original paper.
* Observation normalization followed by clipping to a range (e.g between -10 and 10). This is usually achieved by maintaining a running mean and variance of observations coming from the simulator. [Welford's algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm) is an efficient and numerically stable algorithm for online variance estimation. We have used a Python (Numpy) implementation of Welford's algorithm from [here](https://github.com/a-mitani/welford).
* Normalization of generalized advantage estimates at the batch level.
* Orthogonal initialization of actor, critic networks with appropriate scaling.
* Gradient clipping : ensure that the norm of the concatenated gradients of all parameters does not exceed 0.5
* Early stopping : calculate approximate KL divergence between the current policy and the target, and stop the policy updates of the current epoch if the approximate KL divergence exceeds some preset threshold.
* No weightage given to entropy term in actor loss function.
Some other tricks which we found unnecessary, but which can be found in other PPO implementations are
* Reward scaling
* Clip value loss
  
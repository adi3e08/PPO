# Proximal Policy Optimization (PPO)
A clean and minimal implementation of Proximal Policy Optimization (PPO) algorithm in Pytorch, for continuous action spaces.

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
* Observation normalization followed by clipping to a range (usually between -10 and 10). This is achieved by maintaining a running mean and variance of observations coming from the simulator. For this purpose we use [Welford's algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm), an efficient and numerically stable algorithm for online variance estimation. We use a Numpy implementation of Welford's algorithm from [here](https://github.com/a-mitani/welford).
* Generalized advantage estimation.
* Normalization of generalized advantage estimates at the batch level.
* Orthogonal initialization of actor, critic networks with appropriate scaling.
* Gradient clipping - ensure that the norm of the concatenated gradients of all parameters does not exceed 0.5.
* Early stopping - calculate approximate KL divergence between the current policy and the old policy, and stop the updates of the current epoch if the approximate KL divergence exceeds some preset threshold.
* Zero weightage given to entropy term in actor loss function.
* Separate actor and critic networks.
* tanh activation functions.

Some other tricks that can be found in other PPO implementations, which we found not necessary are,
* Reward scaling - rewards are divided by the standard deviation of a rolling discounted sum of the rewards, followed by clipping to a range (usually between -10 and 10).
* Value function clipping - the value function loss is clipped in a manner that is similar to the PPO???s clipped surrogate objective.
  

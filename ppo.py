import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from welford import Welford
from copy import deepcopy
from functools import partial

def init_weights(module, gain):
    """
    Orthogonal initialization
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

# Joint Actor Critic network
class Pi_V_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Pi_V_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.mu = torch.nn.Linear(64, action_size)
        self.log_sigma = torch.nn.Parameter(torch.zeros(action_size))
        self.v = torch.nn.Linear(64, 1)
        self.distribution = torch.distributions.Normal        
        module_gains = {
            self.fc1: np.sqrt(2),
            self.fc2: np.sqrt(2),
            self.mu: 0.01,
            self.v: 1
        }
        for module, gain in module_gains.items():
            module.apply(partial(init_weights, gain=gain))

    def forward(self, x):
        y1 = torch.tanh(self.fc1(x))
        y2 = torch.tanh(self.fc2(y1))
        mu = self.mu(y2)
        sigma = torch.exp(self.log_sigma)
        dist = self.distribution(mu,sigma)
        values = self.v(y2).view(-1)
        return dist, values

# Actor Network
class Pi_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Pi_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.mu = torch.nn.Linear(64, action_size)
        self.log_sigma = torch.nn.Parameter(torch.zeros(action_size))
        self.distribution = torch.distributions.Normal        
        module_gains = {
            self.fc1: np.sqrt(2),
            self.fc2: np.sqrt(2),
            self.mu: 0.01
        }
        for module, gain in module_gains.items():
            module.apply(partial(init_weights, gain=gain))

    def forward(self, x):
        y1 = torch.tanh(self.fc1(x))
        y2 = torch.tanh(self.fc2(y1))
        mu = self.mu(y2)
        sigma = torch.exp(self.log_sigma)
        dist = self.distribution(mu,sigma)
        return dist

# Critic network
class V_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(V_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.v = torch.nn.Linear(64, 1)
        module_gains = {
            self.fc1: np.sqrt(2),
            self.fc2: np.sqrt(2),
            self.v: 1
        }
        for module, gain in module_gains.items():
            module.apply(partial(init_weights, gain=gain))
            
    def forward(self, x):
        y1 = torch.tanh(self.fc1(x))
        y2 = torch.tanh(self.fc2(y1))
        values = self.v(y2).view(-1)
        return values

def process_dmc_observation(time_step):
    """
    Function to parse observation dictionary returned by Deepmind Control Suite.
    """
    o_1 = np.array([])
    for k in time_step.observation:
        if time_step.observation[k].shape:
            o_1 = np.concatenate((o_1, time_step.observation[k].flatten()))
        else :
            o_1 = np.concatenate((o_1, np.array([time_step.observation[k]])))
    r = time_step.reward
    done = time_step.last()
    return o_1, r, done

def process_observation(x, simulator):
    if simulator == "dm_control":
        o_1, r, done = process_dmc_observation(x)
        if r is None:
            return o_1
        else:
            return o_1, r, done
    elif simulator == "gym":
        if type(x) is np.ndarray:
            return x
        elif type(x) is tuple:
            o_1, r, done, info = x
            return o_1, r, done

# Proximal Policy Optimization Algorithm
class PPO:
    def __init__(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.arglist = parse_args()
        self.env = make_env(seed)
        if self.arglist.use_gpu:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        if self.arglist.normalize_observation:
            self.obs_list = Welford()
        if self.arglist.scale_reward:
            self.r_list = Welford()
            self.r_discounted_sum = 0.0

        if self.arglist.joint_actor_critic:
            self.model = Pi_V_FC(self.env.state_size,self.env.action_size).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.arglist.lr)
        else:
            self.actor = Pi_FC(self.env.state_size,self.env.action_size).to(self.device)
            self.critic = V_FC(self.env.state_size,self.env.action_size).to(self.device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.arglist.lr)

        self.exp_dir = os.path.join("./log", self.arglist.exp_name)
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")
        if os.path.exists("./log"):
            pass            
        else:
            os.mkdir("./log")
        os.mkdir(self.exp_dir)
        os.mkdir(os.path.join(self.tensorboard_dir))
        os.mkdir(self.model_dir)

    def normalize_observation(self, o, train=False):
        if train:
            self.obs_list.add(deepcopy(o))
        if np.isnan(self.obs_list.var_s).any() or (0 in self.obs_list.var_s):
            return np.clip((o-self.obs_list.mean),-10,10)
        else:
            return np.clip((o-self.obs_list.mean)/np.sqrt(self.obs_list.var_s),-10,10)

    def scale_reward(self, r):
        self.r_discounted_sum = r + self.arglist.gamma * self.r_discounted_sum
        self.r_list.add(np.array(self.r_discounted_sum))
        if np.isnan(self.r_list.var_s) or self.r_list.var_s == 0:
            return np.clip(r,-10.0,10.0)
        else:
            return np.clip(r/np.sqrt(self.r_list.var_s),-10.0,10.0)

    def save_checkpoint(self, name):
        checkpoint = {}
        if self.arglist.joint_actor_critic:
            checkpoint['model'] = self.model.state_dict()
        else:
            checkpoint['actor'] = self.actor.state_dict()
        if self.arglist.normalize_observation:
            checkpoint['obs_mean'] = deepcopy(self.obs_list.mean)
            checkpoint['obs_var'] = deepcopy(self.obs_list.var_s)
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        O, A, old_log_probs, old_values, values, R, v_n, gae = [], [], [], [], [], [], [], []
        t = 0
        for episode in range(self.arglist.episodes):
            ep_r = 0.0
            o = process_observation(self.env.reset(), self.env.simulator)
            if self.arglist.normalize_observation:
                o = self.normalize_observation(o, True)
            while True:
                o_ = torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    if self.arglist.joint_actor_critic:
                        dist, values_ = self.model(o_)
                    else:
                        dist = self.actor(o_)
                        values_ = self.critic(o_)
                a = dist.sample()                                   
                old_log_probs.append(dist.log_prob(a).sum())
                old_values.append(values_[0])
                values.append(values_[0])
                O.append(o_[0])
                A.append(a[0])
                o_1, r, done = process_observation(self.env.step(a.cpu().numpy()[0]), self.env.simulator)
                if self.arglist.normalize_observation:
                    o_1 = self.normalize_observation(o_1, True)
                t += 1
                ep_r += r
                if self.arglist.scale_reward:
                    R.append(torch.tensor(self.scale_reward(r), dtype=torch.float, device=self.device))
                else:
                    R.append(torch.tensor(r, dtype=torch.float, device=self.device))
                o = o_1
                if done or t % self.arglist.update_every == 0:
                    with torch.no_grad():
                        if self.arglist.joint_actor_critic: 
                            dist, next_values = self.model(torch.tensor(o_1, dtype=torch.float, device=self.device).unsqueeze(0))
                        else:
                            next_values = self.critic(torch.tensor(o_1, dtype=torch.float, device=self.device).unsqueeze(0))
                    if done and self.env.simulator == "gym":
                        # open ai gym tasks have a terminal state
                        # deep mind control suite tasks are infinite horizon i.e don't have a terminal state
                        values.append(0*next_values[0])
                    else:
                        values.append(next_values[0])
                    v_n_temp = []
                    gae_ = 0
                    gae_temp = []
                    for i in reversed(range(len(R))):
                        gae_ = self.arglist.gamma * self.arglist.gae_lambda * gae_ + R[i] + self.arglist.gamma * values[i+1] - values[i]
                        gae_temp.append(gae_) 
                        v_n_ = gae_ + values[i]
                        v_n_temp.append(v_n_)

                    v_n_temp.reverse()
                    gae_temp.reverse()
                    v_n = v_n + v_n_temp
                    gae = gae + gae_temp
                    values = []
                    R = []

                if t % self.arglist.update_every == 0:

                    O = torch.stack(O)
                    A = torch.stack(A)
                    old_log_probs = torch.stack(old_log_probs)
                    old_values = torch.stack(old_values)
                    v_n = torch.stack(v_n)
                    gae = torch.stack(gae)

                    continue_training = True
                    for epoch in range(self.arglist.epochs):
                        permutation = torch.randperm(O.size(0))
                        for i in range(0, O.size(0), self.arglist.batch_size):
                            indices = permutation[i:i+self.arglist.batch_size]
                            if self.arglist.joint_actor_critic:
                                dist, values = self.model(O[indices])
                            else:
                                dist = self.actor(O[indices])
                                values = self.critic(O[indices])
                            log_probs = dist.log_prob(A[indices]).sum(1)
                            ratio = torch.exp(log_probs-torch.clamp(old_log_probs[indices],min=np.log(1e-5))) # clamp old_prob to 1e-5 to avoid inf

                            with torch.no_grad():
                                approx_kl_div = torch.mean((ratio - 1) - torch.log(ratio)).cpu().numpy()

                            if approx_kl_div > 1.5 * self.arglist.target_kl:
                                continue_training = False
                                break

                            gae_batch = (gae[indices]-gae[indices].mean())/torch.clamp(gae[indices].std(),min=1e-5)
                            surr1 = ratio * gae_batch
                            surr2 = torch.clamp(ratio,1-self.arglist.ppo_clip_term,1+self.arglist.ppo_clip_term) * gae_batch

                            actor_loss = -(torch.min(surr1,surr2) + self.arglist.entropy_weightage * dist.entropy().sum(1)).mean()

                            if self.arglist.use_clipped_value_loss:
                                v_pred = values
                                v_pred_clipped = old_values[indices] + torch.clamp(values - old_values[indices],\
                                                                                   -self.arglist.ppo_clip_term,\
                                                                                   self.arglist.ppo_clip_term)
                                # Unclipped value
                                critic_loss1 = (v_n[indices] - v_pred).pow(2)
                                # Clipped value
                                critic_loss2 = (v_n[indices] - v_pred_clipped).pow(2)
                                critic_loss = 0.5 * torch.max(critic_loss1, critic_loss2).mean()
                            else:
                                critic_loss = 0.5 * (v_n[indices] - values).pow(2).mean()

                            if self.arglist.joint_actor_critic:
                                loss = (actor_loss + self.arglist.critic_loss_weightage * critic_loss)
                                self.optimizer.zero_grad()
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arglist.gradient_clip_term)
                                self.optimizer.step()
                            else:
                                self.critic_optimizer.zero_grad()
                                critic_loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.arglist.gradient_clip_term)
                                self.critic_optimizer.step()

                                self.actor_optimizer.zero_grad()
                                actor_loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.arglist.gradient_clip_term)
                                self.actor_optimizer.step()

                        if not(continue_training):
                            break                             

                    O, A, old_log_probs, old_values, values, R, v_n, gae = [], [], [], [], [], [], [], []
                    t = 0

                if done :
                    writer.add_scalar('train_ep_r', ep_r, episode)
                    if episode % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                        eval_ep_r_list = self.eval(self.arglist.eval_over)
                        writer.add_scalar('eval_ep_r', np.mean(eval_ep_r_list), episode)
                        self.save_checkpoint(str(episode)+".ckpt")
                    break 

    def eval(self, episodes):
        ep_r_list = []
        for episode in range(episodes):
            o = process_observation(self.env.reset(), self.env.simulator)
            if self.arglist.normalize_observation:
                o = self.normalize_observation(o)
            ep_r = 0
            while True:
                with torch.no_grad():
                    if self.arglist.joint_actor_critic:
                        dist, values_ = self.model(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                    else:
                        dist = self.actor(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                a = dist.sample().cpu().numpy()[0]   
                o_1, r, done = process_observation(self.env.step(a), self.env.simulator)
                if self.arglist.normalize_observation:
                    o_1 = self.normalize_observation(o_1)
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    break
        return ep_r_list  

def parse_args():
    parser = argparse.ArgumentParser("Proximal Policy Optimization")
    parser.add_argument("--exp-name", type=str, default="expt_ppo_cartpole_swingup", help="name of experiment")
    parser.add_argument("--episodes", type=int, default=2000, help="number of episodes")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="use gpu")
    # Core training parameters
    parser.add_argument("--joint-actor-critic", action="store_true", default=False, help="joint / separate actor critic")
    parser.add_argument("--normalize-observation", action="store_true", default=True, help="normalize observation")
    parser.add_argument("--use-clipped-value-loss", action="store_true", default=False, help="clip value loss")
    parser.add_argument("--scale-reward", action="store_true", default=False, help="reward scaling")
    parser.add_argument("--target-kl", type=float, default=0.01, help="target kl for early stopping")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument("--gradient-clip-term", type=float, default=0.5, help="clip gradient norm")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='lambda parameter for GAE')
    parser.add_argument('--entropy-weightage', type=float, default=0.0,help='entropy term coefficient')
    parser.add_argument('--critic-loss-weightage', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument("--update-every", type=int, default=2048, help="train after every _ steps")
    parser.add_argument("--eval-every", type=int, default=50, help="eval every _ episodes")
    parser.add_argument("--eval-over", type=int, default=50, help="eval over _ episodes")
    parser.add_argument("--epochs", type=int, default=10, help="ppo epochs")
    parser.add_argument("--ppo-clip-term", type=float, default=0.2, help="ppo clip term")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    return parser.parse_args()

def make_env(env_seed):
    # import gym
    # env = gym.make('BipedalWalker-v3')
    # env.seed(env_seed)
    # env.state_size = 24
    # env.action_size = 4
    # env.action_low = -1
    # env.action_high = 1
    # env.simulator = "gym"

    # from dm_control import suite
    # env = suite.load(domain_name="reacher", task_name="hard", task_kwargs={'random': env_seed})
    # env.state_size = 6
    # env.action_size = 2
    # env.action_low = -1
    # env.action_high = 1
    # env.simulator = "dm_control"

    from dm_control import suite
    env = suite.load(domain_name="cartpole", task_name="swingup", task_kwargs={'random': env_seed})
    env.state_size = 5
    env.action_size = 1
    env.action_low = -1
    env.action_high = 1
    env.simulator = "dm_control"
    
    return env
if __name__ == '__main__':

    ppo = PPO(seed=0)
    ppo.train()
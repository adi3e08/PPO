import os
import argparse
from copy import deepcopy
from collections import deque
from functools import partial
import math
import random
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from welford import Welford
from dm_control import suite
import glob
import subprocess
import matplotlib.pyplot as plt

def init_weights(module, gain):
    """
    Orthogonal initialization
    """
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

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

# Parse observation dictionary returned by Deepmind Control Suite
def process_observation(time_step):
    o_1 = np.array([])
    for k in time_step.observation:
        if time_step.observation[k].shape:
            o_1 = np.concatenate((o_1, time_step.observation[k].flatten()))
        else :
            o_1 = np.concatenate((o_1, np.array([time_step.observation[k]])))
    r = time_step.reward
    done = time_step.last()
    
    return o_1, r, done

# Proximal Policy Optimization algorithm
class PPO:
    def __init__(self, arglist):
        self.arglist = arglist
        
        random.seed(self.arglist.seed)
        np.random.seed(self.arglist.seed)
        torch.manual_seed(self.arglist.seed)
        
        self.env = suite.load(domain_name=self.arglist.domain, task_name=self.arglist.task, task_kwargs={'random': self.arglist.seed})
        obs_spec = self.env.observation_spec()
        action_spec = self.env.action_spec()
        self.obs_size = np.sum([math.prod(obs_spec[k].shape) for k in obs_spec])
        self.action_size = math.prod(action_spec.shape)

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.actor = Pi_FC(self.env.state_size,self.env.action_size).to(self.device)

        if self.arglist.mode == "train":
            self.critic = V_FC(self.env.state_size,self.env.action_size).to(self.device)

            path = "./log/"+self.arglist.domain+"_"+self.arglist.task
            self.exp_dir = os.path.join(path, "seed_"+str(self.arglist.seed))
            self.model_dir = os.path.join(self.exp_dir, "models")
            self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")

            if self.arglist.resume:
                checkpoint = torch.load(os.path.join(self.model_dir,"backup.ckpt"))
                self.start_episode = checkpoint['episode'] + 1

                self.actor.load_state_dict(checkpoint['actor'])
                self.critic.load_state_dict(checkpoint['critic'])

                self.obs_stats = checkpoint['obs_stats']

            else:
                self.start_episode = 0

                self.obs_stats = Welford()

                if not os.path.exists(path):
                    os.makedirs(path)
                os.mkdir(self.exp_dir)
                os.mkdir(self.tensorboard_dir)
                os.mkdir(self.model_dir)

            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.arglist.lr)

            if self.arglist.resume:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

                print("Done loading checkpoint ...")

            self.train()

        elif self.arglist.mode == "eval":
            checkpoint = torch.load(self.arglist.checkpoint,map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.obs_stats = checkpoint['obs_stats']
            ep_r_list = self.eval(self.arglist.episodes,self.arglist.render,self.arglist.save_video)

    def normalize_observation(self, o, update=False):
        if update:
            self.obs_stats.add(deepcopy(o))

        if np.isnan(self.obs_stats.var_s).any() or (0 in self.obs_stats.var_s):
            return np.clip((o-self.obs_stats.mean),-10,10)
        else:
            return np.clip((o-self.obs_stats.mean)/np.sqrt(self.obs_stats.var_s),-10,10)

    def save_checkpoint(self, name):
        checkpoint = {'actor' : self.actor.state_dict(),\
                      'obs_stats' : self.obs_stats \
                      }
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def save_backup(self, episode):
        checkpoint = {'episode' : episode,\
                      'actor' : self.actor.state_dict(),\
                      'actor_optimizer': self.actor_optimizer.state_dict(),\
                      'critic' : self.critic.state_dict(),\
                      'critic_optimizer': self.critic_optimizer.state_dict(),\
                      'obs_stats' : self.obs_stats \
                      }
        torch.save(checkpoint, os.path.join(self.model_dir, "backup.ckpt"))

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        O, A, old_log_probs, old_values, values, R, v_n, gae = [], [], [], [], [], [], [], []
        t = 0
        for episode in range(self.start_episode,self.arglist.episodes):
            ep_r = 0.0
            o,_,_ = process_observation(self.env.reset())
            o = self.normalize_observation(o, True)
            while True:
                o_ = torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    dist = self.actor(o_)
                    values_ = self.critic(o_)
                a = dist.sample()                                   
                old_log_probs.append(dist.log_prob(a).sum())
                old_values.append(values_[0])
                values.append(values_[0])
                O.append(o_[0])
                A.append(a[0])
                o_1, r, done = process_observation(self.env.step(a.cpu().numpy()[0]))
                o_1 = self.normalize_observation(o_1, True)
                t += 1
                ep_r += r
                R.append(torch.tensor(r, dtype=torch.float, device=self.device))
                o = o_1
                if done or t % self.arglist.update_every == 0:
                    with torch.no_grad():
                        next_values = self.critic(torch.tensor(o_1, dtype=torch.float, device=self.device).unsqueeze(0))
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

                            critic_loss = 0.5 * (v_n[indices] - values).pow(2).mean()

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

                if done:
                    writer.add_scalar('ep_r', ep_r, episode)
                    if episode % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                        # Evaluate agent performance
                        eval_ep_r_list = self.eval(self.arglist.eval_over)
                        writer.add_scalar('eval_ep_r', np.mean(eval_ep_r_list), episode)
                        self.save_checkpoint(str(episode)+".ckpt")
                    if (episode % 250 == 0 or episode == self.arglist.episodes-1) and episode > self.start_episode:
                        self.save_backup(episode)
                    break 

    def eval(self, episodes, render=False, save_video=False):
        # Evaluate agent performance over several episodes

        if render and save_video: 
            t = 0
            folder = "./media/"+self.arglist.domain+"_"+self.arglist.task
            subprocess.call(["mkdir","-p",folder])

        ep_r_list = []
        for episode in range(episodes):
            if render:
                vid = None
            o,_,_ = process_observation(self.env.reset())
            o = self.normalize_observation(o)
            ep_r = 0
            while True:
                with torch.no_grad():
                    dist = self.actor(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0))
                a = dist.sample().cpu().numpy()[0] 
                o_1, r, done = process_observation(self.env.step(a))
                o_1 = self.normalize_observation(o_1)
                if render:
                    img = self.env.physics.render(height=240,width=240,camera_id=0)
                    if vid is None:
                        vid = plt.imshow(img)
                    else:
                        vid.set_data(img)
                    plt.axis('off')
                    plt.pause(0.01)
                    plt.draw()
                    if save_video:
                        plt.savefig(folder + "/file%04d.png" % t, bbox_inches='tight')
                        t += 1
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    if render:
                        print("Episode finished with total reward ",ep_r)
                        plt.pause(0.5)                    
                    break        
        if self.arglist.mode == "eval":
            print("Average return :",np.mean(ep_r_list))
            if save_video:
                os.chdir(folder)
                subprocess.call(['ffmpeg', '-i', 'file%04d.png','-r','10','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-pix_fmt', 'yuv420p','video.mp4'])
                for file_name in glob.glob("*.png"):
                    os.remove(file_name)
                subprocess.call(['ffmpeg','-i','video.mp4','video.gif'])
        
        return ep_r_list 

def parse_args():
    parser = argparse.ArgumentParser("PPO")
    # Common settings
    parser.add_argument("--domain", type=str, default="", help="cartpole / reacher")
    parser.add_argument("--task", type=str, default="", help="swingup / hard")
    parser.add_argument("--mode", type=str, default="", help="train or eval")
    parser.add_argument("--episodes", type=int, default=0, help="number of episodes")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    # Core training parameters
    parser.add_argument("--resume", action="store_true", default=False, help="resume training")
    parser.add_argument("--target-kl", type=float, default=0.01, help="target kl for early stopping")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="actor, critic learning rate")
    parser.add_argument("--gradient-clip-term", type=float, default=0.5, help="clip gradient norm")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='lambda parameter for GAE')
    parser.add_argument('--entropy-weightage', type=float, default=0.0,help='entropy term coefficient')
    parser.add_argument('--critic-loss-weightage', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument("--update-every", type=int, default=2048, help="train after every _ steps")
    parser.add_argument("--eval-every", type=int, default=50, help="eval every _ episodes during training")
    parser.add_argument("--eval-over", type=int, default=50, help="each time eval over _ episodes")
    parser.add_argument("--epochs", type=int, default=10, help="ppo epochs")
    parser.add_argument("--ppo-clip-term", type=float, default=0.2, help="ppo clip term")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    # Eval settings
    parser.add_argument("--checkpoint", type=str, default="", help="path to checkpoint")
    parser.add_argument("--render", action="store_true", default=False, help="render")
    parser.add_argument("--save-video", action="store_true", default=False, help="save video")

    return parser.parse_args()

if __name__ == '__main__':
    arglist = parse_args()
    ppo = PPO(arglist)

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_ACTIONS = 3

# set as global variables
PATH = None
FILENAME = None

class NoExistingRoleException(Exception):
    def __init__(self, message="The role chosen for the network must be 'actor' either 'critic'", error_code=404):
        super().__init__(message)
        self.error_code = error_code

    def __str__(self):
        return f"Unexisting role Exception : {self.error_code}. \n {self.message}" 

class simpleCNN(nn.modules):
    def __init__(self, role, n_actions=N_ACTIONS, path=PATH, filename=FILENAME, device=DEVICE):
        
        self.stack_size = 4
        self.frame_stack = []

        self.filepath = os.join.path(path, filename)

        if role == 'actor' or role =='critic':
            self.role = role
        else:
            raise NoExistingRoleException()
        
        self.n_actions = n_actions

        self.conv = nn.Sequential(
            nn.Conv2d(3 * self.stack_size, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3 * self.stack_size, 96, 96)
            conv_out = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_actions),
        )
        
        self.mu_head = nn.Linear(512, self.action_dim)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        self.value_head = nn.Linear(512, 1)
        
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.filepath)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.filepath, map_location=self.device))

    def forward(self, x):
        h = self.conv(x)
        h = self.fc(h)
        mu = self.mu_head(h)
        value = self.value_head(h).squeeze(-1)
        if self.role == "actor":
            dist = torch.distributions.Categorical(torch.distributions.Normal(mu, self.log_std.exp()))
            return dist
        elif self.role == "critic":
            return mu, self.log_std, value
            


def compute_loss(policy_loss, value_loss, value_coeff, entropy, entropy_coeff):
    return policy_loss + value_coeff*value_loss + entropy_coeff*entropy

def compute_GAE(rewards, values, dones, last_value):
    # Fake initialization of hyperparameters
    gamma_fake, lambda_fake = 1, 1
        
    adv = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        next_value = last_value if t == len(rewards) -1 else values[t+1]
        delta = rewards[t] + gamma_fake*next_value*mask - values[t]
        gae = delta + gamma_fake*lambda_fake*mask*gae
        adv[t] = gae
    return adv, adv + values


class PPO(nn.Module):

    def __init__(self, stack_size, device=DEVICE):
        super().__init__()
        self.device = device

        self.stack_size = stack_size
        self.frame_stack = []        

        self.log_prob = None
        self.value_scalar = None
        self.last_state = None
        
        self.actor = simpleCNN(role="actor", device=self.device)
        self.critic = simpleCNN(role="critic", device=self.device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        
        self.to(self.device)
        
    def _preprocess_single(self, obs):
        obs_to_array = np.asarray(obs, dtype=np.float32).transpose(2, 0, 1)
        tensor = torch.tensor(obs_to_array, device=self.device)
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor
    
    def _reset_stack(self, obs):
        frame = self._preprocess_single(obs)
        self.frame_stack = [frame for _ in range(self.stack_size)]
        
    def _get_stacked_obs(self, obs):
        frame = self._preprocess_single(obs)
        self.frame_stack.append(frame)
        if len(self.frame_stack) > self.stack_size:
            self.frame_stack.pop(0)
        stacked = torch.cat(self.frame_stack, dim=0)
        return stacked.unsqueeze(0)
    
    def act(self, state):
        if len(self.frame_stack) == 0:
            self._reset_stack(state)
        
        x = self._get_stacked_obs(state)
        self.last_state = x
        
        distribution = self.actor(state)
        value = self.critic(state)
        action = distribution.sample()
        
        # Study the dimensionality of these elements and if needed to take an item of action or it's already an action
        self.log_prob = distribution.log_prob(action).item()
        self.value_scalar = value.item()
        action = action.item()
        
        return action
        
        
        
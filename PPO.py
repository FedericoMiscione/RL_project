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

'''
The input sequence should be a 7x7x3 representation of the grid in which each coordinate (x, y)
of the grid contains a vector [v_1, v_2, v_3] in which:
- Channel 0 -> represents the type of the object (Empty, Wall, Lava, Goal)
- Channel 1 -> represents the color of the object as an integer
- Channel 2 -> represents the state of the object

Interesting point: Channels' values are categories, then could be interesting to use a embedding
layer before the CNN in order to transform these indices in dense vectors.
'''
class simpleCNN(nn.Module):
    def __init__(self, role, n_actions=N_ACTIONS, path=PATH, filename=FILENAME, device=DEVICE):
        
        self.stack_size = 4
        self.frame_stack = []

        self.filepath = os.path.join(path, filename)

        if role == 'actor' or role =='critic':
            self.role = role
        else:
            raise NoExistingRoleException()
        
        self.n_actions = n_actions

        # Using the RGBImgPartialObsWrapper provided by Gymnasium the input will appear as an image of size 56x56
        # otherwise let's continue with a 7x7 image
        self.conv = nn.Sequential(
            nn.Conv2d(3 * self.stack_size, 16, 3, stride=1), # It's for 96x96 images but we have Minigrid 7x7
            nn.ReLU(),
            nn.Conv2d(16, 32, 1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3 * self.stack_size, 7, 7)
            conv_out = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 256),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(256, self.n_actions)
        self.value_head = nn.Linear(256, 1)
        
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.filepath)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.filepath, map_location=self.device))

    def forward(self, x):
        h = self.conv(x)
        h = self.fc(h)
                
        if self.role == "actor":
            return torch.distributions.Categorical(logits=self.policy_head(h))
        elif self.role == "critic":
            return self.value_head(h).squeeze(-1)
            
def compute_loss(policy_loss, value_loss, value_coeff, entropy, entropy_coeff):
    return policy_loss + value_coeff*value_loss + entropy_coeff*entropy

def compute_GAE(rewards, values, dones, last_value, gamma_, lambda_):
    adv = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        next_value = last_value if t == len(rewards) -1 else values[t+1]
        delta = rewards[t] + gamma_*next_value*mask - values[t]
        gae = delta + gamma_*lambda_*mask*gae
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
        
    def update(self, n_steps, env):
        obs, _ = env.reset
        self._reset_stack(obs)
        
        states_vec, actions_vec, old_logprob_vec, values_vec, rewards_vec, dones_vec = [], [], [], [], [], []
        episodes_rewards = []
        
        episode_reward = 0.0
        
        for _ in range(n_steps):
            
            action = self.act(obs)
            
            states_vec.append(self.last_state.squeeze(0).cpu().numpy())
            actions_vec.append(action)
            old_logprob_vec.append(self.log_prob)
            values_vec.append(self.value_scalar)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            rewards_vec.append(reward)
            dones_vec.append(done)
            
            episode_reward += reward
            
            if done:
                episodes_rewards.append()
                episode_reward = 0.0
                obs, _ = env.reset()
                self._reset_stack(obs)
                
        return obs, states_vec, actions_vec, old_logprob_vec, rewards_vec, values_vec, dones_vec, episodes_rewards
        
        
    def learn(self, n_updates, n_steps, gamma_, lambda_, clip_epsilon, minibatch, value_coeff, entropy_coeff):
        env = gym.make("MiniGrid-LavaGapS7-v0", render_mode="human")
        
        for update in range(n_updates):
            obs, states, actions, old_logprob, rewards, values, dones, episodes_rewards = self.update(n_steps, env)
            
            with torch.no_grad():
                last_value = self.critic(self._get_stacked_obs(obs)).item()
            
            advantages, returns = compute_GAE(np.array(rewards), np.array(values), np.array(dones, dtype=np.float32), last_value, gamma_, lambda_)
            
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            advantages = (advantages - advantages.mean())/ (advantages.std() + 1e-8)
            states_tensor = torch.stack([torch.tensor(state, device=self.device, dtype=torch.float32) for state in np.array(states)])
            action_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
            old_logprob_tensor = torch.tensor(old_logprob, dtype=torch.float32, device=self.device)
            advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            indices = np.arange(len(np.array(states)))
            surrogate_values = []
            
            # Here there was a loop for ppo_epochs
            np.random.shuffle(indices)
            for start in range(0, len(np.array(states)), minibatch):
                batch = indices[start:start + minibatch]
                s = states_tensor[batch]
                a = action_tensor[batch]
                old_lp = old_logprob_tensor[batch]
                A = advantages_tensor[batch]
                R = returns_tensor[batch]
                
                distribution = self.actor(s)
                logp = distribution.log_prob(a)
                
                v_pred = self.critic(s)
                
                ratio = torch.exp(logp - old_lp)
                
                surrogate1 = ratio * A
                surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * A
                policy_loss = - torch.min(surrogate1, surrogate2).mean()
                
                value_loss = F.mse_loss(v_pred, R)
                entropy = distribution.entropy().sum(-1).mean()
                
                loss = compute_loss(policy_loss=policy_loss,
                                    value_loss=value_loss, value_coeff=value_coeff,
                                    entropy=entropy, entropy_coeff=entropy_coeff
                                    )
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                
                surrogate_values.append(policy_loss.item())
                
            mean_reward = np.mean(episodes_rewards) if episodes_rewards else 0.0
            print(f"[PPO] Update {update+1}/{n_updates} completed | Mean episode reward: {mean_reward:.2f}")
            
        print("Training completed.")
        
    # Probably removable
    def save(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        
    def load(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn as nn
import csv
import codecs
from collections import deque
import random
import numpy as np
import pathlib
from torch.distributions import Categorical

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (string): GPU or CPU
        """

        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
  

class DQNAgent():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model, load_model, model_name):
        """Initialize an Agent object.
        
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            buffer_size (int): replay buffer size
            batch_size (int):  minibatch size
            gamma (float): discount factor
            lr (float): learning rate 
            update_every (int): how often to update the network
            replay_after (int): After which replay to be started
            model(Model): Pytorch Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.replay_after = replay_after
        self.DQN = model
        self.tau = tau
        self.root_dir = ""

        
        # Q-Network
        # self.policy_net = self.DQN(num_actions = action_size).to(self.device)
        # self.target_net = self.DQN(num_actions = action_size).to(self.device)
        self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        self.target_net = self.DQN(input_shape, action_size).to(self.device)
        
        # self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        # self.target_net = self.DQN(input_shape, action_size).to(self.device)

        if load_model:
            self.load_model(self.root_dir, model_name[0], self.policy_net)
            self.load_model(self.root_dir, model_name[1], self.target_net)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)
        
        self.t_step = 0
    
    def load_model(self, root, model_name, model):
        filepath = pathlib.Path(root + model_name + '.pt')
        if filepath.exists():
            model.load_state_dict(torch.load(str(filepath)))
            print(model_name + " loaded!")

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        # print(state)
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def act_test(self, state, eps=0.):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        # self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from policy model
        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0]
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)
            

    def save_networks(self, curr_episode, fin_episode):
        self.save_every = 100
        self.root_dir = "model/"
        if curr_episode % self.save_every == 0:
            self.save_model(self.root_dir, "mario-ep-{}_policy_net".format(str(curr_episode)), self.policy_net)
            self.save_model(self.root_dir, "mario-ep-{}_target_net".format(str(curr_episode)), self.target_net)
        if curr_episode == fin_episode:
            self.save_model(self.root_dir, "mario_policy_net_fin", self.policy_net)
            self.save_model(self.root_dir, "mario_target_net_fin", self.target_net)
    
    def save_model(self, root, model_name, model):
        filepath = pathlib.Path(root + model_name + '.pt')
        if not filepath.parent.exists():
            filepath.parent.mkdir()
        torch.save(model.state_dict(), str(filepath))
        print(model_name + " saved!")

class DDQNAgent():
    def __init__(self, input_shape, action_size, seed, device, buffer_size, batch_size, gamma, lr, tau, update_every, replay_after, model):
        """Initialize an Agent object.
        
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            buffer_size (int): replay buffer size
            batch_size (int):  minibatch size
            gamma (float): discount factor
            lr (float): learning rate 
            update_every (int): how often to update the network
            replay_after (int): After which replay to be started
            model(Model): Pytorch Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_every = update_every
        self.replay_after = replay_after
        self.DQN = model
        self.tau = tau

        
        # Q-Network
        self.policy_net = self.DQN(num_actions = action_size).to(self.device)
        self.target_net = self.DQN(num_actions = action_size).to(self.device)
        # self.policy_net = self.DQN(input_shape, action_size).to(self.device)
        # self.target_net = self.DQN(input_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed, self.device)
        
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.replay_after:
                experiences = self.memory.sample()
                self.learn(experiences)
                
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from policy model
        Q_expected_current = self.policy_net(states)
        Q_expected = Q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_net(next_states).detach().max(1)[0]
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    
    # θ'=θ×τ+θ'×(1−τ)
    def soft_update(self, policy_model, target_model, tau):
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)
            
    def save_networks(self, curr_episode, fin_episode):
        self.save_every = 100
        self.root_dir = "model/"
        if curr_episode % self.save_every == 0:
            self.save_model(self.root_dir, "mario-ep-{}_policy_net".format(str(curr_episode)), self.policy_net)
            self.save_model(self.root_dir, "mario-ep-{}_target_net".format(str(curr_episode)), self.target_net)
        if curr_episode == fin_episode:
            self.save_model(self.root_dir, "mario_policy_net_fin", self.policy_net)
            self.save_model(self.root_dir, "mario_target_net_fin", self.target_net)
    
    def save_model(self, root, model_name, model):
        filepath = pathlib.Path(root + model_name + '.pt')
        if not filepath.parent.exists():
            filepath.parent.mkdir()
        torch.save(model.state_dict(), str(filepath))
        print(model_name + " saved!")


            

class DQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
class DDQNCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DDQNCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage  - advantage.mean()
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

class ActorCnn(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCnn, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        dist = Categorical(x)
        return dist
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

class CriticCnn(nn.Module):
    def __init__(self, input_shape):
        super(CriticCnn, self).__init__()
        self.input_shape = input_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    
    
    
class A2CAgent():
    def __init__(self, input_shape, action_size, seed, device, gamma, alpha, beta, update_every, actor_m, critic_m):
        """Initialize an Agent object.
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            gamma (float): discount factor
            alpha (float): Actor learning rate
            beta (float): Critic learning rate 
            update_every (int): how often to update the network
            actor_m(Model): Pytorch Actor Model
            critic_m(Model): PyTorch Critic Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.update_every = update_every

        # Actor-Network
        self.actor_net = actor_m(input_shape, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)

        # Critic-Network
        self.critic_net = critic_m(input_shape).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)

        # Memory
        self.log_probs = []
        self.values    = []
        self.rewards   = []
        self.masks     = []
        self.entropies = []

        self.t_step = 0

    def step(self, state, log_prob, entropy, reward, done, next_state):

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        value = self.critic_net(state)
        
        # Save experience in  memory
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(torch.from_numpy(np.array([reward])).to(self.device))
        self.masks.append(torch.from_numpy(np.array([1 - done])).to(self.device))
        self.entropies.append(entropy)

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
           self.learn(next_state)
           self.reset_memory()
                
    def act(self, state):
        """Returns action, log_prob, entropy for given state as per current policy."""
        
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_probs = self.actor_net(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        entropy = action_probs.entropy().mean()

        return action.item(), log_prob, entropy

        
        
    def learn(self, next_state):
        next_state = torch.from_numpy(next_state).unsqueeze(0).to(self.device)
        next_value = self.critic_net(next_state)

        returns = self.compute_returns(next_value, self.gamma)

        log_probs = torch.cat(self.log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(self.values)

        advantage = returns - values

        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * sum(self.entropies)

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def reset_memory(self):
        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.masks[:]
        del self.entropies[:]

    def compute_returns(self, next_value, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
        return returns

    def save_networks(self, curr_episode, fin_episode):
        self.save_every = 100
        self.root_dir = "model/"
        if curr_episode % self.save_every == 0:
            self.save_model(self.root_dir, "mario-ep-{}_actor_net".format(str(curr_episode)), self.actor_net)
            self.save_model(self.root_dir, "mario-ep-{}_critic_net".format(str(curr_episode)), self.critic_net)
        if curr_episode == fin_episode:
            self.save_model(self.root_dir, "mario_actor_net_fin", self.actor_net)
            self.save_model(self.root_dir, "mario_critic_net_fin", self.critic_net)
    
    def save_model(self, root, model_name, model):
        filepath = pathlib.Path(root + model_name + '.pt')
        if not filepath.parent.exists():
            filepath.parent.mkdir()
        torch.save(model.state_dict(), str(filepath))
        print(model_name + " saved!")


class PPOAgent():
    def __init__(self, input_shape, action_size, seed, device, gamma, alpha, beta, tau, update_every, batch_size, ppo_epoch, clip_param, actor_m, critic_m, load_model, model_name):
        """Initialize an Agent object.
        Params
        ======
            input_shape (tuple): dimension of each state (C, H, W)
            action_size (int): dimension of each action
            seed (int): random seed
            device(string): Use Gpu or CPU
            gamma (float): discount factor
            alpha (float): Actor learning rate
            beta (float): Critic learning rate 
            tau (float): Tau Value
            update_every: How often to update network
            batch_size (int): Mini Batch size to be used every epoch 
            ppo_epoch(int): Total No epoch for ppo
            clip_param(float): Clip Paramter
            actor_m(Model): Pytorch Actor Model
            critic_m(Model): PyTorch Critic Model
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = device
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.ppo_epoch = ppo_epoch
        self.clip_param = clip_param
        self.root_dir = ""

        # Actor-Network
        self.actor_net = actor_m(input_shape, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.alpha)

        # Critic-Network
        self.critic_net = critic_m(input_shape).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.beta)

        
        if load_model:
            self.load_model(self.root_dir, model_name[0], self.actor_net)
            self.load_model(self.root_dir, model_name[1], self.critic_net)
        
        # Memory
        self.log_probs = []
        self.values    = []
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.masks     = []
        self.entropies = []

        self.t_step = 0
        
    def load_model(self, root, model_name, model):
        filepath = pathlib.Path(root + model_name + '.pt')
        if filepath.exists():
            model.load_state_dict(torch.load(str(filepath)))
            print(model_name + " loaded!")

    def step(self, state, action, value, log_prob, reward, done, next_state):
        
        # Save experience in  memory
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.states.append(torch.from_numpy(state).unsqueeze(0).to(self.device))
        self.rewards.append(torch.from_numpy(np.array([reward])).to(self.device))
        self.actions.append(torch.from_numpy(np.array([action])).to(self.device))
        self.masks.append(torch.from_numpy(np.array([1 - done])).to(self.device))

        self.t_step = (self.t_step + 1) % self.update_every

        if self.t_step == 0:
           self.learn(next_state)
           self.reset_memory()
                
    def act(self, state):
        """Returns action, log_prob, value for given state as per current policy."""
        
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_probs = self.actor_net(state)
        value = self.critic_net(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)

        return action.item(), log_prob, value
    
    def act_test(self, state, eps=0.):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_probs = self.actor_net(state)
        value = self.critic_net(state)

        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)

        return action.item(), log_prob, value
        
    def learn(self, next_state):
        next_state = torch.from_numpy(next_state).unsqueeze(0).to(self.device)
        next_value = self.critic_net(next_state)

        returns        = torch.cat(self.compute_gae(next_value)).detach()
        self.log_probs = torch.cat(self.log_probs).detach()
        self.values    = torch.cat(self.values).detach()
        self.states    = torch.cat(self.states)
        self.actions   = torch.cat(self.actions)
        advantages     = returns - self.values

        for _ in range(self.ppo_epoch):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(returns, advantages):

                dist = self.actor_net(state)
                value = self.critic_net(state)

                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()
                
                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

                # Minimize the loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.reset_memory()

    
    def ppo_iter(self, returns, advantage):
        memory_size = self.states.size(0)
        for _ in range(memory_size // self.batch_size):
            rand_ids = np.random.randint(0, memory_size, self.batch_size)
            yield self.states[rand_ids, :], self.actions[rand_ids], self.log_probs[rand_ids], returns[rand_ids, :], advantage[rand_ids, :]

    def reset_memory(self):
        self.log_probs = []
        self.values    = []
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.masks     = []
        self.entropies = []

    def compute_gae(self, next_value):
        gae = 0
        returns = []
        values = self.values + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + self.gamma * self.tau * self.masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
    
    def save_networks(self, curr_episode, fin_episode):
        self.save_every = 100
        self.root_dir = "model/"
        if curr_episode % self.save_every == 0:
            self.save_model(self.root_dir, "mario-ep-{}_actor_net".format(str(curr_episode)), self.actor_net)
            self.save_model(self.root_dir, "mario-ep-{}_critic_net".format(str(curr_episode)), self.critic_net)
        if curr_episode == fin_episode:
            self.save_model(self.root_dir, "mario_actor_net_fin", self.actor_net)
            self.save_model(self.root_dir, "mario_critic_net_fin", self.critic_net)
    
    def save_model(self, root, model_name, model):
        filepath = pathlib.Path(root + model_name + '.pt')
        if not filepath.parent.exists():
            filepath.parent.mkdir()
        torch.save(model.state_dict(), str(filepath))
        print(model_name + " saved!")
        
class DQNCnn_Sonic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNCnn_Sonic, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            # nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU()
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU()
            # nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

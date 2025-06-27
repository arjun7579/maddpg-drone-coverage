import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# --- Hyperparameters for 5 drones ---
GRID_SIZE = 20
DRONE_COUNT = 5
USER_COUNT = 10
OBSTACLE_COUNT = 5
COVERAGE_RADIUS = 3
OBS_DIM = 2 + USER_COUNT*2 + OBSTACLE_COUNT*4
STATE_DIM = DRONE_COUNT*2 + USER_COUNT*2 + OBSTACLE_COUNT*4
ACTION_DIM = 2  # continuous dx, dy
BATCH_SIZE = 128
BUFFER_SIZE = 50000
GAMMA = 0.90
TAU = 0.05
LR_ACTOR = 2e-3
LR_CRITIC = 2e-3
EPISODES = 1000
STEPS_PER_EP = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# --- Environment ---
class UAVEnv:
    def __init__(self):
        self.reset()
    def reset(self):
        self.drones = [np.random.randint(0, GRID_SIZE, 2) for _ in range(DRONE_COUNT)]
        self.users = [np.random.randint(0, GRID_SIZE, 2) for _ in range(USER_COUNT)]
        self.obstacles = [self._random_obstacle() for _ in range(OBSTACLE_COUNT)]
        self.steps = 0
        return self._get_obs(), self._get_state()
    def _random_obstacle(self):
        pos = np.random.randint(0, GRID_SIZE, 2)
        dynamic = random.random() < 0.7
        direction = random.choice([(0,1),(0,-1),(1,0),(-1,0)]) if dynamic else (0,0)
        speed = random.uniform(0.5, 1.5) if dynamic else 0.0
        return {'pos': pos, 'dynamic': dynamic, 'direction': direction, 'speed': speed}
    def _move_obstacles(self):
        for obs in self.obstacles:
            if obs['dynamic']:
                x, y = obs['pos']
                dx, dy = obs['direction']
                nx = x + dx * obs['speed']
                ny = y + dy * obs['speed']
                if not (0 <= nx < GRID_SIZE): dx *= -1; nx = x + dx * obs['speed']
                if not (0 <= ny < GRID_SIZE): dy *= -1; ny = y + dy * obs['speed']
                obs['direction'] = (dx, dy)
                obs['pos'] = np.array([nx, ny])
    def _get_obs(self):
        obs_n = []
        for d in self.drones:
            obs = list(np.array(d)/GRID_SIZE)
            for u in self.users: obs.extend(np.array(u)/GRID_SIZE)
            for o in self.obstacles:
                obs.extend(o['pos']/GRID_SIZE)
                obs.append(float(o['dynamic']))
                obs.append(o['speed']/2.0)
            obs_n.append(np.array(obs, dtype=np.float32))
        return obs_n
    def _get_state(self):
        vec = []
        for d in self.drones: vec.extend(np.array(d)/GRID_SIZE)
        for u in self.users: vec.extend(np.array(u)/GRID_SIZE)
        for o in self.obstacles:
            vec.extend(o['pos']/GRID_SIZE)
            vec.append(float(o['dynamic']))
            vec.append(o['speed']/2.0)
        return np.array(vec, dtype=np.float32)
    def step(self, actions):
        self._move_obstacles()
        obs_pos = [obs['pos'] for obs in self.obstacles]
        obs_radii = 1.0
        new_drones = []
        rewards = []
        for i, act in enumerate(actions):
            x, y = self.drones[i]
            dx = float(act[0]) * 1.5
            dy = float(act[1]) * 1.5
            nx = np.clip(x + dx, 0, GRID_SIZE-1)
            ny = np.clip(y + dy, 0, GRID_SIZE-1)
            collided = False
            for op in obs_pos:
                if np.linalg.norm([nx-op[0], ny-op[1]]) < obs_radii:
                    collided = True
                    break
            if collided:
                nx, ny = x, y
            new_drones.append([nx, ny])
        self.drones = new_drones
        for i, (dx, dy) in enumerate(self.drones):
            covered = 0
            for ux, uy in self.users:
                if np.hypot(dx-ux, dy-uy) <= COVERAGE_RADIUS:
                    covered += 1
            rewards.append(covered)
        self.steps += 1
        done = self.steps >= STEPS_PER_EP
        return self._get_obs(), self._get_state(), rewards, done, {}

# --- MADDPG Core ---
class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + n_agents*action_dim, 384)
        self.fc2 = nn.Linear(384, 384)
        self.fc3 = nn.Linear(384, 1)
    def forward(self, state, actions):
        x = torch.cat([state, actions.view(actions.size(0), -1)], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MADDPGAgent:
    def __init__(self, obs_dim, state_dim, action_dim, n_agents):
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.actor_target = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim, n_agents).to(device)
        self.critic_target = Critic(state_dim, action_dim, n_agents).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
    def select_action(self, obs, noise=0.1):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy().flatten()
        action += noise * np.random.randn(*action.shape)
        return np.clip(action, -1, 1)
    def update(self, replay_buffer, agents, agent_idx):
        if len(replay_buffer) < BATCH_SIZE: return
        obs_b, state_b, actions_b, rewards_b, next_obs_b, next_state_b, dones_b = replay_buffer.sample(BATCH_SIZE)
        obs_b = np.array(obs_b)
        actions_b = np.array(actions_b)
        rewards_b = np.array(rewards_b)
        next_obs_b = np.array(next_obs_b)
        next_state_b = np.array(next_state_b)
        dones_b = np.array(dones_b)
        obs = torch.FloatTensor(obs_b[:,agent_idx,:]).to(device)
        state = torch.FloatTensor(state_b).to(device)
        actions = torch.FloatTensor(actions_b).to(device)
        rewards = torch.FloatTensor(rewards_b[:,agent_idx]).unsqueeze(1).to(device)
        next_obs = torch.FloatTensor(next_obs_b[:,agent_idx,:]).to(device)
        next_state = torch.FloatTensor(next_state_b).to(device)
        dones = torch.FloatTensor(dones_b).unsqueeze(1).to(device)
        next_actions = []
        for i, ag in enumerate(agents):
            next_actions.append(ag.actor_target(torch.FloatTensor(next_obs_b[:,i,:]).to(device)))
        next_actions = torch.stack(next_actions, dim=1)
        with torch.no_grad():
            q_next = self.critic_target(next_state, next_actions)
            q_target = rewards + GAMMA * q_next * (1 - dones)
        q_val = self.critic(state, actions)
        critic_loss = nn.MSELoss()(q_val, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        curr_actions = [ag.actor(torch.FloatTensor(obs_b[:,i,:]).to(device)) if i==agent_idx else torch.FloatTensor(actions_b[:,i,:]).to(device) for i,ag in enumerate(agents)]
        curr_actions = torch.stack(curr_actions, dim=1)
        actor_loss = -self.critic(state, curr_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for t, s in zip(self.actor_target.parameters(), self.actor.parameters()):
            t.data.copy_(TAU*s.data + (1-TAU)*t.data)
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.copy_(TAU*s.data + (1-TAU)*t.data)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, obs, state, actions, rewards, next_obs, next_state, done):
        self.buffer.append((obs, state, actions, rewards, next_obs, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, state, actions, rewards, next_obs, next_state, dones = map(np.array, zip(*batch))
        return obs, state, actions, rewards, next_obs, next_state, dones
    def __len__(self):
        return len(self.buffer)

# --- Training Loop ---
import os

def train():
    env = UAVEnv()
    agents = [MADDPGAgent(OBS_DIM, STATE_DIM, ACTION_DIM, DRONE_COUNT) for _ in range(DRONE_COUNT)]
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    # Create directory for saving models
    os.makedirs("saved_models", exist_ok=True)

    for ep in range(EPISODES):
        obs_n, state = env.reset()
        total_rewards = np.zeros(DRONE_COUNT)
        for step in range(STEPS_PER_EP):
            actions = [agent.select_action(obs) for agent, obs in zip(agents, obs_n)]
            next_obs_n, next_state, rewards, done, _ = env.step(actions)
            replay_buffer.push(obs_n, state, actions, rewards, next_obs_n, next_state, done)
            obs_n, state = next_obs_n, next_state
            total_rewards += np.array(rewards)
            for i, agent in enumerate(agents):
                agent.update(replay_buffer, agents, i)
            if done: break

        print(f"Episode {ep+1:4d}: Drone rewards: {total_rewards}")

        # Save every 500 episodes
        if (ep + 1) % 500 == 0:
            for i, agent in enumerate(agents):
                torch.save(agent.actor.state_dict(), f"saved_models/actor_agent_{i}_ep{ep+1}.pth")
                torch.save(agent.critic.state_dict(), f"saved_models/critic_agent_{i}_ep{ep+1}.pth")
            print(f"Saved models at episode {ep+1}")


if __name__ == "__main__":
    train()

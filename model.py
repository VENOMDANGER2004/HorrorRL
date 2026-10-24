import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Force PyTorch to use CPU for lightning-fast inference
device = torch.device("cpu")


# ==========================================
# 1. DQN (Deep Q-Network)
# Best for: Tracking fading noise and generalizing paths.
# ==========================================
class GhostDQN(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=4):
        super(GhostDQN, self).__init__()
        # Tuned Architecture: Wider hidden layers (128) to process noise intensity accurately
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Outputs Q-Values for Up, Down, Left, Right


# ==========================================
# 2. PPO / Direct RL (Actor-Critic)
# Best for: Natural, unpredictable 3D movement.
# ==========================================
class GhostPPO(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, output_size=4):
        super(GhostPPO, self).__init__()
        # Shared feature extractor
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # Actor: Outputs the probability of choosing each direction (Policy)
        self.actor = nn.Linear(hidden_size, output_size)
        # Critic: Estimates how "good" the current state is (Value)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.base(x)
        action_probs = F.softmax(self.actor(features), dim=-1)
        state_value = self.critic(features)
        return action_probs, state_value


# ==========================================
# 3. Q-LEARNING (Tabular Memory)
# Best for: Rapid learning on static map layouts.
# ==========================================
class GhostQLearning:
    def __init__(self, actions=4, alpha=0.1, gamma=0.99):
        self.q_table = {}  # Dynamic dictionary to handle state-space
        self.actions = actions
        self.alpha = alpha  # Learning Rate
        self.gamma = gamma  # Discount Factor

    def _discretize_state(self, state):
        # State: [GhostX, GhostY, NoiseX, NoiseY, NoiseIntensity]
        # We must round the noise intensity so the table doesn't grow infinitely
        gx, gy, nx, ny, intensity = state
        return (int(gx), int(gy), int(nx), int(ny), round(intensity, 1))

    def predict(self, state):
        state_key = self._discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.actions)
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state):
        state_key = self._discretize_state(state)
        next_key = self._discretize_state(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.actions)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(self.actions)

        # Bellman Equation Update for Tabular Q-Learning
        best_next_q = np.max(self.q_table[next_key])
        self.q_table[state_key][action] += self.alpha * (
                    reward + self.gamma * best_next_q - self.q_table[state_key][action])


# ==========================================
# 4. MDP (Pure Mathematical Value Iteration)
# Best for: Flawless, terminator-style pathfinding (Wallhack fallback).
# ==========================================
class GhostMDP:
    def __init__(self, grid_size=10, walls=None):
        self.grid_size = grid_size
        self.walls = walls if walls else set()
        self.gamma = 0.95

    def predict(self, ghost_pos, noise_pos, intensity):
        """Mathematically floods the map with values based on noise location."""
        if intensity == 0.0:
            # If no noise, patrol randomly or stay put
            return np.random.randint(0, 4)

        gx, gy = int(ghost_pos[0]), int(ghost_pos[1])
        nx, ny = int(noise_pos[0]), int(noise_pos[1])

        V = np.zeros((self.grid_size, self.grid_size))
        V[nx, ny] = 100 * intensity  # Reward scales with how loud the noise was

        # Run Bellman sweeps to find optimal path around walls
        for _ in range(15):
            new_V = np.copy(V)
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if (x, y) == (nx, ny) or (x, y) in self.walls:
                        continue
                    vals = []
                    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Up, Down, Left, Right
                        adj_x, adj_y = x + dx, y + dy
                        if 0 <= adj_x < self.grid_size and 0 <= adj_y < self.grid_size and (adj_x,
                                                                                            adj_y) not in self.walls:
                            vals.append(V[adj_x, adj_y])
                        else:
                            vals.append(0)
                    new_V[x, y] = max(vals) * self.gamma
            V = new_V

        # Pick best adjacent tile
        max_val = -float('inf')
        best_action = 0
        actions = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

        for action, (dx, dy) in actions.items():
            adj_x, adj_y = gx + dx, gy + dy
            if 0 <= adj_x < self.grid_size and 0 <= adj_y < self.grid_size and (adj_x, adj_y) not in self.walls:
                if V[adj_x, adj_y] > max_val:
                    max_val = V[adj_x, adj_y]
                    best_action = action

        return best_action


if __name__ == "__main__":
    print("✅ model.py successfully loaded. All 4 AI Brains are ready for training.")
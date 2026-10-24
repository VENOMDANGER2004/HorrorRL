import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from environment import HorrorEnvironment
from model import GhostDQN

# ==========================================
# HYPERPARAMETERS & SETUP
# ==========================================
BATCH_SIZE = 64
GAMMA = 0.99  # Discount factor (long-term planning)
EPSILON_START = 1.0  # 100% random exploration at start
EPSILON_END = 0.05  # 5% random exploration at the end
EPSILON_DECAY = 0.995  # Epsilon decay rate
LEARNING_RATE = 0.001
EPISODES = 1000  # Training rounds

# Initialize the world and the brain
env = HorrorEnvironment()
model = GhostDQN(input_size=5, hidden_size=128, output_size=4)  # Using the hyper-tuned 128 hidden layer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Memory buffer (Experience Replay)
memory = deque(maxlen=10000)


def train_model():
    print("🧠 Starting Deep Q-Network Training on CPU...")
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:  # 100 step limit to prevent infinite wandering
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

                # Take the step
            next_state, reward, done = env.step_ghost(action)

            # Save to memory
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            steps += 1

            # --- BACKPROPAGATION ---
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)

                b_states = torch.FloatTensor(np.array([b[0] for b in batch]))
                b_actions = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
                b_rewards = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1)
                b_next_states = torch.FloatTensor(np.array([b[3] for b in batch]))
                b_dones = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1)

                current_q = model(b_states).gather(1, b_actions)

                with torch.no_grad():
                    max_next_q = model(b_next_states).max(1)[0].unsqueeze(1)
                    target_q = b_rewards + GAMMA * max_next_q * (1 - b_dones)

                loss = loss_fn(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{EPISODES} | Avg Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f} | Steps: {steps}")

    torch.save(model.state_dict(), "ghost_dqn_brain.pth")
    print("✅ Training Complete! The Ghost's consciousness is saved as 'ghost_dqn_brain.pth'")


if __name__ == "__main__":
    train_model()
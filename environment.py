import numpy as np


class HorrorEnvironment:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size

        # Define the map layout (Walls)
        self.walls = {(2, 3), (3, 2), (4, 5), (5, 4), (7, 7), (8, 7), (2, 8), (7, 2)}

        # Entities
        self.player_pos = [9, 9]
        self.ghost_pos = [0, 0]

        # --- THE NOISE SYSTEM (POMDP) ---
        # The AI no longer knows exactly where the player is.
        # It only knows where a sound happened and how loud it was.
        self.last_noise = {'x': 9, 'y': 9, 'intensity': 0.0}

    def reset(self):
        """Resets the board for a new training episode or new game."""
        self.player_pos = [self.grid_size - 1, self.grid_size - 1]
        self.ghost_pos = [0, 0]
        self.last_noise = {'x': self.player_pos[0], 'y': self.player_pos[1], 'intensity': 0.0}
        return self._get_ai_state()

    def _get_ai_state(self):
        """
        Translates the game world into the 5-number array the AI actually 'sees'.
        State = [Ghost X, Ghost Y, Noise X, Noise Y, Noise Intensity]
        """
        return np.array([
            self.ghost_pos[0],
            self.ghost_pos[1],
            self.last_noise['x'],
            self.last_noise['y'],
            self.last_noise['intensity']
        ], dtype=np.float32)

    def move_player(self, action, movement_type="walk"):
        """
        Handles player movement and generates sound based on how they move.
        0: Up, 1: Down, 2: Left, 3: Right
        movement_type: 'sprint' (Loud), 'walk' (Medium), 'sneak' (Silent)
        """
        px, py = self.player_pos

        if action == 0 and py > 0 and (px, py - 1) not in self.walls:
            py -= 1
        elif action == 1 and py < self.grid_size - 1 and (px, py + 1) not in self.walls:
            py += 1
        elif action == 2 and px > 0 and (px - 1, py) not in self.walls:
            px -= 1
        elif action == 3 and px < self.grid_size - 1 and (px + 1, py) not in self.walls:
            px += 1

        self.player_pos = [px, py]

        # Generate Sound Events
        if movement_type == "sprint":
            self.last_noise = {'x': px, 'y': py, 'intensity': 1.0}
        elif movement_type == "walk":
            self.last_noise = {'x': px, 'y': py, 'intensity': 0.5}
        # If 'sneak', intensity doesn't increase, keeping the player hidden!

    def step_ghost(self, action):
        """
        Executes the AI's chosen move, decays noise, and calculates the reward.
        """
        gx, gy = self.ghost_pos

        # Ghost Movement Physics
        if action == 0 and gy > 0 and (gx, gy - 1) not in self.walls:
            gy -= 1
        elif action == 1 and gy < self.grid_size - 1 and (gx, gy + 1) not in self.walls:
            gy += 1
        elif action == 2 and gx > 0 and (gx - 1, gy) not in self.walls:
            gx -= 1
        elif action == 3 and gx < self.grid_size - 1 and (gx + 1, gy) not in self.walls:
            gx += 1

        self.ghost_pos = [gx, gy]

        # Noise Decay over time (The sound fades away)
        self.last_noise['intensity'] = max(0.0, self.last_noise['intensity'] - 0.1)

        # Check Win/Loss Condition
        caught = (self.ghost_pos == self.player_pos)

        # Reward System for AI Training
        if caught:
            reward = 100.0  # Massive reward for a kill
        else:
            # Small penalty to encourage hunting quickly
            # Extra penalty if it wanders while noise is loud
            reward = -1.0 - (self.last_noise['intensity'] * 0.5)

        return self._get_ai_state(), reward, caught


# Quick test to ensure PyCharm reads it correctly
if __name__ == "__main__":
    env = HorrorEnvironment()
    print("Initial AI State:", env.reset())
import os
import platform
import random
import math
import numpy as np
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from environment import HorrorEnvironment
from model import GhostQLearning, GhostMDP

# ==========================================
# 🩹 WINDOWS PYTORCH DLL PRE-LOAD HOTFIX
# ==========================================
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec

    try:
        if (spec := find_spec("torch")) and spec.origin:
            dll_path = os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
            if os.path.exists(dll_path):
                ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

# ==========================================
# 🎛️ CONFIGURATION PANEL
# ==========================================
ACTIVE_MODEL = "MDP"
DIFFICULTY = "NORMAL"

# Ensure this EXACTLY matches the texture image file you placed in the folder!
SIREN_TEXTURE = "siren_skin.png"
# ==========================================

TORCH_AVAILABLE = False
try:
    import torch
    from model import GhostDQN, GhostPPO

    TORCH_AVAILABLE = True
except Exception:
    pass

app = Ursina(borderless=False)
window.fps_counter.enabled = True
window.exit_button.visible = False

# ==========================================
# 🌅 "FEARS TO FATHOM" ATMOSPHERE
# ==========================================
window.color = color.hex('#110502')
Sky(color=color.hex('#7a200d'))

scene.fog_density = 0.04
scene.fog_color = color.hex('#360d04')

# ==========================================
# 3D ENVIRONMENT BUILD
# ==========================================
grid_env = HorrorEnvironment()
TILE_SCALE = 3.0

ground = Entity(model='plane', scale=(100, 1, 100), color=color.hex('#1a1a1a'), texture='white_cube',
                texture_scale=(20, 20), collider='box')

for wall in grid_env.walls:
    Entity(model='cube', position=Vec3(wall[0] * TILE_SCALE, 1.5, wall[1] * TILE_SCALE),
           scale=(TILE_SCALE, 3.5, TILE_SCALE), color=color.hex('#333333'), texture='brick', collider='box')

player = FirstPersonController(position=Vec3(9 * TILE_SCALE, 2, 9 * TILE_SCALE), speed=6)
camera.fov = 90

# ==========================================
# 🧟‍♂️ THE REAL 3D SIREN HEAD LOADER (CLEAN OBJ)
# ==========================================
# Loads your freshly cleaned obj file and paints the texture over it
ghost_wrapper = Entity(
    model='sirenhead.obj',
    texture=SIREN_TEXTURE,
    color=color.white,
    scale=300,
    position=Vec3(0, 0, 0),
    collider='box'
)

# Map-wide audio setting
# Map-wide background horror audio (Changed to .ogg for compatibility)
static_audio = Audio('static.ogg', loop=True, autoplay=True, volume=1.0, pitch=1.0)

# ==========================================
# 📼 UI & JUMPSCARE OVERLAYS
# ==========================================
rec_dot = Entity(parent=camera.ui, model='circle', color=color.red, scale=(0.02, 0.02), position=(-0.84, 0.45))
rec_text = Text(text="REC", position=(-0.81, 0.45), scale=2, color=color.red)
time_text = Text(text="PM 09:10:13", position=(0.6, 0.45), scale=1.5, color=color.white)

minimap_bg = Entity(parent=camera.ui, model='quad', color=color.black66, scale=(0.25, 0.25), position=(-0.7, 0.2))
minimap_dots = {}

for x in range(10):
    for y in range(10):
        pos_x = -0.7 + (x - 4.5) * 0.022
        pos_y = 0.2 + (y - 4.5) * 0.022
        if (x, y) in grid_env.walls:
            Entity(parent=camera.ui, model='quad', color=color.rgba(255, 255, 255, 50), scale=(0.02, 0.02),
                   position=(pos_x, pos_y))
        else:
            minimap_dots[(x, y)] = Entity(parent=camera.ui, model='quad', color=color.black, scale=(0.018, 0.018),
                                          position=(pos_x, pos_y))

debug_text = Text(text="", position=(-0.85, -0.35), scale=1.2, color=color.white)

death_screen = Entity(parent=camera.ui, model='quad', color=color.rgba(180, 0, 0, 235), scale=(2, 2), visible=False)
death_eyes = Text(text="O_O", scale=25, position=(-0.15, 0.1), color=color.black, visible=False)
restart_text = Text(text="SIGNAL LOST\nPRESS 'R' TO RESTART\nMade By Vaibhav and Harsh", position=(-0.25, -0.2), scale=2, color=color.white,
                    visible=False)

# ==========================================
# GAME STATE & AI INITIALIZATION
# ==========================================
if not TORCH_AVAILABLE and ACTIVE_MODEL in ["DQN", "PPO"]:
    ACTIVE_MODEL = "MDP"

dqn_brain, ppo_brain = None, None
if TORCH_AVAILABLE:
    dqn_brain = GhostDQN(input_size=5, hidden_size=128, output_size=4)
    ppo_brain = GhostPPO(input_size=5, hidden_size=128, output_size=4)
    if ACTIVE_MODEL == "DQN":
        try:
            dqn_brain.load_state_dict(torch.load("ghost_dqn_brain.pth")); dqn_brain.eval()
        except:
            pass

q_brain = GhostQLearning(actions=4)
mdp_brain = GhostMDP(grid_size=10, walls=grid_env.walls)

speed_multipliers = {"EASY": 0.6, "NORMAL": 1.0, "HARD": 1.3, "NIGHTMARE": 1.7}
GHOST_SPEED = speed_multipliers[DIFFICULTY]

# ==========================================
# MAIN GAME LOOP
# ==========================================
ai_clock = 0.0
ai_update_interval = 0.35
target_3d_position = Vec3(0, 0, 0)
flicker_clock = 0.0
is_dead = False


def update():
    global ai_clock, target_3d_position, flicker_clock, is_dead

    if is_dead:
        return

    # VHS UI Blinking
    flicker_clock += time.dt
    if flicker_clock > 0.8:
        rec_dot.visible = not rec_dot.visible
        rec_text.visible = not rec_text.visible
        flicker_clock = 0.0

    movement_mode = "walk"
    noise_val = 0.5
    player.speed = 6

    if held_keys['left shift']:
        movement_mode = "sprint"
        noise_val = 1.0
        player.speed = 9
    elif held_keys['left control']:
        movement_mode = "sneak"
        noise_val = 0.0
        player.speed = 3

    grid_px = clamp(int(round(player.x / TILE_SCALE)), 0, 9)
    grid_py = clamp(int(round(player.z / TILE_SCALE)), 0, 9)
    grid_env.player_pos = [grid_px, grid_py]

    if movement_mode != "sneak":
        grid_env.last_noise = {'x': grid_px, 'y': grid_py, 'intensity': noise_val}

    debug_text.text = f"BRAIN: {ACTIVE_MODEL} | LVL: {DIFFICULTY}\nMIC: {movement_mode.upper()}"

    # AI Logic Tiks
    ai_clock += time.dt
    if ai_clock >= ai_update_interval:
        state = np.array([grid_env.ghost_pos[0], grid_env.ghost_pos[1],
                          grid_env.last_noise['x'], grid_env.last_noise['y'],
                          grid_env.last_noise['intensity']], dtype=np.float32)
        action = 0

        if ACTIVE_MODEL == "MDP":
            gamma_map = {"EASY": 0.30, "NORMAL": 0.70, "HARD": 0.95, "NIGHTMARE": 0.99}
            mdp_brain.gamma = gamma_map[DIFFICULTY]
            action = mdp_brain.predict(grid_env.ghost_pos, [grid_env.last_noise['x'], grid_env.last_noise['y']],
                                       grid_env.last_noise['intensity'])
        elif ACTIVE_MODEL == "Q_LEARNING":
            action = random.randint(0, 3) if random.random() < 0.15 else q_brain.predict(state)
        elif TORCH_AVAILABLE:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            if ACTIVE_MODEL == "DQN":
                with torch.no_grad(): action = torch.argmax(dqn_brain(state_t)).item()

        grid_env.step_ghost(action)
        target_3d_position = Vec3(grid_env.ghost_pos[0] * TILE_SCALE, 0, grid_env.ghost_pos[1] * TILE_SCALE)
        ai_clock = 0.0

    # Smooth Movement & Tracking
    ghost_wrapper.position = lerp(ghost_wrapper.position, target_3d_position, time.dt * 4.0 * GHOST_SPEED)
    ghost_wrapper.look_at(player.position, axis='forward')

    distance_to_player = (ghost_wrapper.position - player.position).length()

    # --- PROCEDURAL RUNNING ANIMATION ---
    if distance_to_player > 3.5:
        animate_speed = 14.0 * GHOST_SPEED
        ghost_wrapper.y = math.sin(time.time() * animate_speed) * 0.35
        ghost_wrapper.rotation_z = math.cos(time.time() * animate_speed) * 12.0
    else:
        ghost_wrapper.y = 0
        ghost_wrapper.rotation_z = 0

    # --- DYNAMIC AUDIO SYSTEM ---
    if static_audio and static_audio.clip is not None:
        if distance_to_player < 100:
            static_audio.volume = max(0.15, 1.0 - (distance_to_player / 100.0))
            if distance_to_player < 25:
                static_audio.pitch = 1.0 + ((25.0 - distance_to_player) / 8.0)
            else:
                static_audio.pitch = 1.0
        else:
            static_audio.volume = 0

    # Update Minimap Indicators
    for coord, dot in minimap_dots.items():
        if list(coord) == grid_env.ghost_pos:
            dot.color = color.red
        elif list(coord) == [grid_px, grid_py]:
            dot.color = color.green
        elif coord[0] == grid_env.last_noise['x'] and coord[1] == grid_env.last_noise['y'] and grid_env.last_noise[
            'intensity'] > 0:
            dot.color = color.yellow
        else:
            dot.color = color.black

    # --- JUMPSCARE TRIGGER ---
    if distance_to_player < 3.5:
        trigger_jumpscare()


def trigger_jumpscare():
    global is_dead
    is_dead = True
    player.speed = 0

    if static_audio and static_audio.clip is not None:
        static_audio.volume = 0

    death_screen.visible = True
    death_eyes.visible = True
    restart_text.visible = True
    debug_text.text = ""


def input(key):
    global is_dead
    if key == 'r' and is_dead:
        restart_game()


def restart_game():
    global is_dead, target_3d_position

    grid_env.ghost_pos = [0, 0]
    target_3d_position = Vec3(0, 0, 0)
    ghost_wrapper.position = Vec3(0, 0, 0)
    ghost_wrapper.y = 0
    ghost_wrapper.rotation_z = 0
    grid_env.last_noise = {'x': 9, 'y': 9, 'intensity': 0.0}

    player.position = Vec3(9 * TILE_SCALE, 2, 9 * TILE_SCALE)
    player.speed = 6

    death_screen.visible = False
    death_eyes.visible = False
    restart_text.visible = False
    is_dead = False


app.run()
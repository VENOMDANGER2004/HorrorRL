from ursina import *

app = Ursina()

# 1. AUDIO TEST
test_audio = Audio('static.wav', loop=True, autoplay=True)

# 2. REFERENCE CUBE (Turned into a wireframe so we can see inside it)
reference_cube = Entity(model='cube', color=color.red, position=(0, 0, 10), scale=2, wireframe=True)

# 3. THE MONSTER (Forced to be double-sided in case it's inside-out)
monster = Entity(
    model='sirenhead.obj',
    color=color.white,
    position=(0, 0, 10),
    double_sided=True
)

# --- THE AUTO-SCALER (FIXED) ---
print("\n" + "=" * 50)
if monster.model:
    print(f"[✓] Model Loaded!")

    # Calculate the scale size by subtracting the bounding box corners
    size_x = abs(monster.bounds.end[0] - monster.bounds.start[0])
    size_y = abs(monster.bounds.end[1] - monster.bounds.start[1])
    size_z = abs(monster.bounds.end[2] - monster.bounds.start[2])
    original_size = max(size_x, size_y, size_z)

    print(f"[*] Calculated Physical Size (X, Y, Z): ({size_x:.4f}, {size_y:.4f}, {size_z:.4f})")
    print(f"[*] Original Max Dimension: {original_size:.4f}")

    if original_size > 0:
        # Force the model to be exactly 2 units tall inside our reference box
        monster.scale = 2.0 / original_size
        print(f"[*] Forced New Scale Multiplier: {monster.scale:.6f}")
    else:
        print("[X] ERROR: The model dimensions are 0. The mesh might be empty.")
else:
    print(f"[X] CRITICAL: Model failed to load.")
print("=" * 50 + "\n")

EditorCamera()
Text(text="Scroll to Zoom. Right-Click to Look. WASD to Move.", position=(-0.5, 0.4))

app.run()
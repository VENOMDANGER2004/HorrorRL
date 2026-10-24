import trimesh

print("Loading rigged GLB...")
# Make sure your original 11MB or 2MB GLB file is named sirenhead.glb
scene = trimesh.load('sirenhead.glb')

print("Stripping skeleton and generating clean OBJ...")
scene.export('sirenhead.obj')

print("Done! You can now run your game.")
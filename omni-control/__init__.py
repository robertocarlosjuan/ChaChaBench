# omni-control/__init__.py
import omnigibson as og
from omnigibson.macros import gm
import os

print("Initializing omni-control package: Applying global OmniGibson settings...")

# --- OmniGibson Global Settings ---
# These settings are applied once when the omni-control package is first imported.
gm.HEADLESS = False # Run headless
gm.REMOTE_STREAMING = "webrtc" # or None or "native"
gm.RENDER_VIEWER_CAMERA = True

# Set paths (modify if necessary)
# IMPORTANT: Ensure these paths are correct for the environment where this code runs
gm.ASSET_PATH = "/nethome/che321/flash/camera-motion/OmniData/assets"
gm.DATASET_PATH = "/nethome/che321/flash/camera-motion/OmniData/og_dataset"
gm.KEY_PATH = "/nethome/che321/flash/camera-motion/OmniData/omnigibson.key"

# You could add checks here to ensure paths exist if desired
if not os.path.exists(gm.ASSET_PATH):
    print(f"WARNING: Configured ASSET_PATH does not exist: {gm.ASSET_PATH}")
if not os.path.exists(gm.DATASET_PATH):
    print(f"WARNING: Configured DATASET_PATH does not exist: {gm.DATASET_PATH}")
# Add similar check for KEY_PATH if necessary

print("Global OmniGibson settings applied.") 
# omnig-control/config.py
#omnigibson is in previous directory
# import sys
# sys.path.append("../omnigibson")
import omnigibson as og
from omnigibson.macros import gm
import os
import math
from typing import List, Tuple

class SceneConfiguration:
    """Holds configuration settings for the scene and camera control."""
    def __init__(self, scene_model: str = "Rs_int"):
        # --- Scene & Video ---
        self.scene_model: str = scene_model # Default scene, can be overridden
        self.video_dir: str = "/nethome/che321/flash/camera-motion/OmniData/videos/scene_nav" # Base dir for non-generated videos
        # Video filename/path might be more specific to the run, handled by generator?
        # self.video_filename: str = f"{self.scene_model}_camera_control.mp4"
        # self.video_path: str = os.path.join(self.video_dir, self.video_filename)

        # Optional custom map path - generator might handle this selection
        self.custom_map_path: str = f"/nethome/che321/flash/camera-motion/OmniData/og_dataset/scenes/{self.scene_model}/layout/floor_trav_0-new.png"
        self.floor_num_to_use: int = 0 # Which floor map to use/modify

        # --- Camera Control ---
        self.camera_z_height: float = 1.0 # Initial height above floor
        self.move_meters_per_sec: float = 0.5
        self.turn_rad_per_sec: float = math.pi / 4 # 45 deg/sec
        self.sim_fps: int = 30
        self.safety_radius: float = 0.1 # Meters for map erosion
        self.max_sampling_attempts: int = 50 # For initial pose sampling
        self.max_precheck_attempts: int = 50 # Max attempts to find start point + safe path
        self.max_movement_amount_tries: int = 20 # Max attempts to find valid movement amount

        # --- Initial Pose ---
        # Example: Look along +Y world axis (quat for 90 deg rot around Z)
        self.initial_orientation: List[float] = [0.7071, 0.0, 0.0, 0.7071]
        # Example: Look along +X world axis (identity)
        # self.initial_orientation: List[float] = [0.0, 0.0, 0.0, 1.0]
        # Example: Look down -Z world axis
        # self.initial_orientation: List[float] = [0.7071, 0.0, 0.0, 0.7071] # Your previous value


        # --- Command Sequence ---
        # Format: (type: "move" or "turn", direction/axis: str, amount: float)
        self.commands: List[Tuple[str, str, float]] = [
            ("move", "forward", 0.5),
            ("move", "backward", 0.5),
            ("move", "left", 0.5),
            ("move", "right", 0.5),
            ("move", "up", 0.5),
            ("move", "down", 0.5),
        ]
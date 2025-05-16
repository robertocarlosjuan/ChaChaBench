# omnig-control/controller.py
import omnigibson as og
from omnigibson.macros import gm

from omnigibson.maps.traversable_map import TraversableMap

import cv2
import os
import numpy as np
import torch as th
import math
from typing import List, Tuple, Optional, Any, TYPE_CHECKING

# Type hint for Numpy Array
NumpyArray = np.ndarray # Define alias for clarity
# Type hint for Position and Orientation tuple
PosOri = Tuple[th.Tensor, th.Tensor]

# Use TYPE_CHECKING to avoid circular imports if needed, though unlikely here
if TYPE_CHECKING:
    # This allows forward references without quotes if classes were defined later
    pass

# Use relative imports assuming files are in the same package/directory
from .config import SceneConfiguration
from .utils import (
    MOVE_DIRECTION_TO_AXIS, TURN_AXIS_TO_AXIS,
    quat_multiply, quat_from_axis_angle, rotate_vector_by_quat
)
from .recorder import VideoRecorder # <-- Import the new recorder class

class SceneCameraController:
    """Handles scene setup, camera control, pre-checking, and recording."""

    def __init__(self, config: SceneConfiguration):
        """Initializes the controller."""
        self.config: SceneConfiguration = config
        self.env: Optional[og.Environment] = None
        self.scene: Optional[og.Scene] = None
        self.camera: Optional[Any] = None # VisionSensor, but avoid strict typing if import is complex
        self.recorder: VideoRecorder = VideoRecorder() # <-- Instantiate recorder
        self.trav_map_obj: Optional[TraversableMap] = None
        self.eroded_map: Optional[th.Tensor] = None
        self.map_shape: Optional[Tuple[int, int]] = None
        self.total_steps_taken: int = 0
        self.initial_pos: Optional[th.Tensor] = None
        self.initial_ori: Optional[th.Tensor] = None

    # --- Setup Methods ---

    def setup(self) -> bool:
        """Performs all setup steps."""
        if not self._setup_environment(): return False
        if not self._load_and_process_custom_map(): return False # Optional step
        # if not self._find_valid_start_pose_with_precheck(): return False
        return True

    def _setup_environment(self) -> bool:
        """Creates the OmniGibson environment."""
        print("*" * 20 + " CONFIGURING ENVIRONMENT " + "*" * 20)
        # Config should have already applied gm settings
        cfg = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": self.config.scene_model,
                "default_erosion_radius": self.config.safety_radius,
            },
            "robots": [{"type": "Fetch", "name": "dummy_robot", "obs_modalities": [], "visible": False}],
        }
        print(f"\ncfg: {cfg}\n---")
        try:
 # Ensure viewer camera is requested
            self.env = og.Environment(configs=cfg)
            self.scene = self.env.scene
            self.camera = og.sim.viewer_camera
            self.trav_map_obj = self.scene.trav_map if self.scene else None

            if self.camera is None: raise RuntimeError("Viewer camera not available.")
            if self.trav_map_obj is None: raise RuntimeError("TraversableMap not available.")
            print(f"Loaded Scene: {self.config.scene_model}, Using Viewer Camera")
            return True
        except Exception as e:
            print(f"Error creating environment: {e}")
            return False

    def _load_and_process_custom_map(self) -> bool:
        """Loads, processes, and injects a custom traversability map if specified."""
        if not self.trav_map_obj: return False # Should be caught by env setup
        if not self.config.custom_map_path:
            print("No custom map path specified. Using default map.")
            return True # Not an error
        
        print(f"---\nUsing custom map: {self.config.custom_map_path}\n---")

        print("*" * 20 + " LOADING CUSTOM TRAVERSABILITY MAP " + "*" * 20)
        floor_num = self.config.floor_num_to_use

        if os.path.exists(self.config.custom_map_path):
            try:
                custom_map_img = cv2.imread(self.config.custom_map_path, cv2.IMREAD_GRAYSCALE)
                if custom_map_img is None:
                    raise IOError(f"cv2 could not read {self.config.custom_map_path}")

                map_size = self.trav_map_obj.map_size
                if map_size is None:
                     raise ValueError("Map size not available from trav_map_obj.")

                print(f"  Resizing custom map to {map_size}x{map_size}")
                resized_map_np = cv2.resize(custom_map_img, (map_size, map_size)) 
                resized_map_tensor = th.tensor(resized_map_np, dtype=th.uint8)

                processed_map = self._process_map_image(resized_map_tensor)
                cv2.imwrite(f"processed_map_{self.config.scene_model}_floor_{floor_num}.png", processed_map.cpu().numpy())
                print(f"  Processed map saved to processed_map_{self.config.scene_model}_floor_{floor_num}.png\n")
                print(f" len(self.trav_map_obj.floor_map): {len(self.trav_map_obj.floor_map)}")

                if floor_num < len(self.trav_map_obj.floor_map):
                    self.trav_map_obj.floor_map[floor_num] = processed_map
                    print(f"  Successfully replaced floor {floor_num} map data.")
                    if hasattr(self.trav_map_obj, '_eroded_map_cache'):
                        self.trav_map_obj._eroded_map_cache = {}
                else:
                    print(f"  Warning: Floor index {floor_num} out of bounds.")
            except Exception as e:
                print(f"Error loading/processing custom map {self.config.custom_map_path}: {e}")
                print("  Proceeding with the default map.")
                # Don't return False, just proceed with default
        else:
            print(f"Custom map path not found: {self.config.custom_map_path}. Using default.")
        return True

    def _process_map_image(self, map_tensor: th.Tensor) -> th.Tensor:
        """Converts map image tensor to binary (0 or 255)."""
        processed_map = map_tensor.clone()
        # Assumes white (255) is traversable, everything else becomes non-traversable (0)
        processed_map[processed_map < 255] = 0
        return processed_map

    def _generate_eroded_map(self) -> bool:
        """Generates the eroded map for pre-checks."""
        if not self.trav_map_obj: return False
        floor_num = self.config.floor_num_to_use
        print(f"Generating eroded map for floor {floor_num}...")

        if floor_num >= len(self.trav_map_obj.floor_map):
            print("Error: Floor index invalid.")
            return False

        base_map = self.trav_map_obj.floor_map[floor_num]
        # Use the object's method which uses the default_erosion_radius (set to safety_radius)
        self.eroded_map = self.trav_map_obj._erode_trav_map(th.clone(base_map))
        self.map_shape = self.eroded_map.shape
        print(f"Eroded map generated using default radius {self.trav_map_obj.default_erosion_radius}m. Shape: {self.map_shape}")
        # Optional: Save for debugging
        # cv2.imwrite("eroded_map_for_precheck.png", self.eroded_map.cpu().numpy())
        return True

    def _find_valid_start_pose_with_precheck(self, commands: List[Tuple[str, str, float]]=None) -> bool:
        """Loops sampling and pre-checking until a valid start pose is found."""
        print("*"*20 + " FINDING VALID INITIAL POSE & PRE-CHECKING PATH " + "*"*20)
        if not self._generate_eroded_map(): return False # Need map first

        for attempt in range(self.config.max_precheck_attempts):
            print(f"Attempt {attempt + 1}/{self.config.max_precheck_attempts}...")
            start_pose = self._sample_start_pose()
            if start_pose is None:
                print("  Failed to sample a point in this attempt.")
                continue # Try again

            pos, ori = start_pose
            if self._pre_check_traversal(pos, ori, commands):
                self.initial_pos = pos
                self.initial_ori = ori
                print(f"Found valid start pose and passed pre-check: {pos.tolist()}")
                # Set the camera pose now that it's validated
                self.camera.set_position_orientation(self.initial_pos, self.initial_ori)
                og.sim.step() # Step once to sync pose
                print(f"Initial Orientation: {self.initial_ori.tolist()}")
                return True
            else:
                print("  Pre-check failed for this starting position. Re-sampling...")

        print(f"Error: Failed to find valid start pose/path after {self.config.max_precheck_attempts} attempts.")
        return False

    def _sample_start_pose(self) -> Optional[PosOri]:
        """Samples a single potential starting pose."""
        if not self.scene or not self.trav_map_obj: return None
        floor_num = self.config.floor_num_to_use

        for _ in range(self.config.max_sampling_attempts):
            _, sampled_pos = self.scene.get_random_point(floor=floor_num)
            if sampled_pos is not None:
                pos = sampled_pos + th.tensor([0.0, 0.0, self.config.camera_z_height], dtype=th.float32)
                ori = th.tensor(self.config.initial_orientation, dtype=th.float32)
                print(f"  Sampled potential start (X,Y,Z): {pos.tolist()}")
                return pos, ori
        return None # Failed to sample after attempts

    # --- Pre-Check Logic ---

    def _pre_check_traversal(self, start_pos: th.Tensor, start_ori: th.Tensor, commands: List[Tuple[str, str, float]]=None) -> bool:
        """Checks if the XY path of the command sequence is likely safe."""
        if self.eroded_map is None or self.trav_map_obj is None or self.map_shape is None:
            print("Error: Pre-check cannot run without eroded map.")
            return False

        if commands is None:
            print("Error: No commands provided to pre-check.")
            return False

        print("  Performing traversal pre-check...")
        current_pos = start_pos.clone()
        current_quat = start_ori.clone()

        # Check start position
        if not self._is_xy_safe(current_pos[:2]):
            print("    Pre-check failed: Initial position invalid on eroded map.")
            return False

        for cmd_idx, (cmd_type, direction_or_axis, amount) in enumerate(commands):
            if cmd_type == "turn":
                current_quat = self._simulate_turn(current_quat, direction_or_axis, amount)
                if current_quat is None: continue # Unknown axis warning already printed
            elif cmd_type == "move":
                target_pos = self._calculate_target_pos(current_pos, current_quat, direction_or_axis, amount)
                if target_pos is None: continue # Unknown direction warning

                # Check path only for horizontal movements (relative to camera)
                if direction_or_axis not in ["up", "down"]:
                    if not self._is_path_segment_safe(current_pos, target_pos, cmd_idx):
                        return False # Failure message printed inside helper

                current_pos = target_pos # Update position for next command check
            else:
                print(f"    Pre-check warning: Unknown command type '{cmd_type}'")

        print("  Traversal pre-check successful.")
        return True

    def _simulate_turn(self, current_quat: th.Tensor, axis: str, amount: float) -> Optional[th.Tensor]:
        """Calculates the orientation after a turn command."""
        if axis not in TURN_AXIS_TO_AXIS:
            print(f"    Pre-check warning: Unknown turn axis '{axis}'")
            return None
        local_axis = TURN_AXIS_TO_AXIS[axis]
        delta_quat = quat_from_axis_angle(local_axis.tolist(), amount)
        new_quat = quat_multiply(current_quat, delta_quat)
        return new_quat / th.linalg.norm(new_quat)

    def _calculate_target_pos(self, current_pos: th.Tensor, current_quat: th.Tensor, direction: str, amount: float) -> Optional[th.Tensor]:
        """Calculates the target position after a move command."""
        if direction not in MOVE_DIRECTION_TO_AXIS:
            print(f"    Pre-check warning: Unknown move direction '{direction}'")
            return None
        local_dir = MOVE_DIRECTION_TO_AXIS[direction]
        world_disp = rotate_vector_by_quat(local_dir * amount, current_quat)
        return current_pos + world_disp

    def _is_xy_safe(self, world_xy: th.Tensor) -> bool:
        """Checks if a single world XY coordinate is safe on the eroded map."""
        if self.eroded_map is None or self.trav_map_obj is None or self.map_shape is None:
             return False
        map_xy = self.trav_map_obj.world_to_map(world_xy).round().long()
        r, c = map_xy[0], map_xy[1]
        if not (0 <= r < self.map_shape[0] and 0 <= c < self.map_shape[1]):
            return False # Out of bounds
        
        return self.eroded_map[r, c].item() != 0 # Safe if not 0

    def _is_path_segment_safe(self, start_pos: th.Tensor, target_pos: th.Tensor, cmd_idx: int) -> bool:
        """Checks intermediate interpolated points of a horizontal move segment, form start to target."""
        if self.eroded_map is None or self.trav_map_obj is None: return False
        world_disp_total = target_pos - start_pos
        amount = th.linalg.norm(world_disp_total[:2]) # Use XY distance for step calc? Or total? Let's use total.
        amount_total = th.linalg.norm(world_disp_total)
        duration = abs(amount_total.item()) / self.config.move_meters_per_sec # Use total distance
        num_command_steps = max(1, int(round(duration * self.config.sim_fps)))

        for j in range(num_command_steps):
            interp_factor = (j + 1) / num_command_steps
            interp_pos = start_pos + world_disp_total * interp_factor
            if not self._is_xy_safe(interp_pos[:2]):
                 map_xy = self.trav_map_obj.world_to_map(interp_pos[:2]).round().long()
                 print(f"    Pre-check failed: Step {j+1}/{num_command_steps} of command {cmd_idx+1} hits unsafe area at map({map_xy[0]},{map_xy[1]}).")
                 return False
        return True

    # --- Simulation Execution ---

    def run_simulation(self):
        """Executes the command sequence in the simulator."""
        if not self.camera or not self.eroded_map:
            print("Error: Controller not fully initialized for simulation.")
            return

        # Start recorder using config path
        if not self.config.video_path: # Check if a path is defined in config
             print("Error: config.video_path not set for run_simulation.")
             return
        record_success = self.recorder.start(
            output_path=self.config.video_path, # Assuming config defines this
            fps=self.config.sim_fps,
            frame_width=self.camera.image_width,
            frame_height=self.camera.image_height
        )

        try:
            if not record_success:
                print("Error: Failed to start recorder for run_simulation.")
                return

            print("*"*20 + " RUNNING SIMULATION (Command Control - Config Sequence) " + "*"*20)
            num_commands = len(self.config.commands)
            for cmd_idx, command in enumerate(self.config.commands):
                print(f"Executing Command {cmd_idx + 1}/{num_commands}: {command[0]} {command[1]} {command[2]:.2f}")
                # Execute using the same helper as the data generator uses
                success = self._execute_single_command_sim(command)
                if not success:
                    print(f"Error or safety stop during command {cmd_idx + 1}.")
                    break
            print("Finished config command sequence.")

        finally:
            self.recorder.release() # Ensure recorder is released

    def _execute_single_command_sim(self, command: Tuple[str, str, float], start_duration: float=0.0) -> bool:
        """Executes one command, steps sim, tells recorder to write frames."""
        cmd_type, direction_or_axis, amount = command
        current_pos, current_quat = self.camera.get_position_orientation()

        # Calculate target pose
        target_pos = current_pos
        target_quat = current_quat
        is_safe_move = True # Less critical due to pre-check, but good fallback

        if cmd_type == "move":
            sim_target_pos = self._calculate_target_pos(current_pos, current_quat, direction_or_axis, amount)
            if sim_target_pos is None: return True # Skip unknown direction
            target_pos = sim_target_pos
            # Safety check target pos on XY plane
            if not self._is_xy_safe(target_pos[:2]):
                 map_xy = self.trav_map_obj.world_to_map(target_pos[:2]).round().long()
                 print(f"  SIM LOOP WARNING: Target unsafe ({map_xy[0]},{map_xy[1]}). Skipping command.")
                 is_safe_move = False
                 print(f" SIM LOOP WARNING: Target unsafe ({map_xy[0]},{map_xy[1]}). Skipping command.")
                 return False, None # Skip command safely
        elif cmd_type == "turn":
             sim_target_quat = self._simulate_turn(current_quat, direction_or_axis, amount)
             if sim_target_quat is None: return True, None # Skip unknown axis
             target_quat = sim_target_quat
        else:
            print(f" SIM LOOP Warning: Unknown command type '{cmd_type}'. Skipping.")
            return False, None # Treat skip as handled

        if not is_safe_move:
            return False, None # Skip command safely

        # Calculate steps
        if cmd_type == "move": duration = abs(amount) / self.config.move_meters_per_sec
        elif cmd_type == "turn": duration = abs(amount) / self.config.turn_rad_per_sec
        num_command_steps = max(1, int(round(duration * self.config.sim_fps)))

        # Execute steps
        curr_duration = start_duration
        duration2pose = []
        start_pos, start_quat = current_pos, current_quat
        for j in range(num_command_steps):
            interp_factor = (j + 1) / num_command_steps
            interp_pos, interp_quat = self._interpolate_pose(start_pos, start_quat, target_pos, target_quat, cmd_type, interp_factor)
            self.camera.set_position_orientation(interp_pos, interp_quat)

            curr_duration += 1.0 / self.config.sim_fps
            # Step sim and TELL recorder to write frame
            if not self._step_and_record_frame(): # Renamed slightly
                print(f" SIM LOOP WARNING: Step and record frame failed. Skipping command.")
                return False, None # Stop simulation if recording fails

        #store the end pose
        duration2pose.append({'duration': curr_duration, 'pose': interp_pos.tolist(), 'orientation': interp_quat.tolist()})

        return True, duration2pose

    def _interpolate_pose(self, start_p, start_q, target_p, target_q, cmd_type, factor) -> PosOri:
        """Interpolates pose between start and target."""
        if cmd_type == "move":
            interp_pos = start_p + (target_p - start_p) * factor
            interp_quat = start_q # Orientation constant
        elif cmd_type == "turn":
            interp_pos = start_p # Position constant
            # Simple LERP + normalization
            interp_quat = start_q * (1.0 - factor) + target_q * factor
            interp_quat = interp_quat / th.linalg.norm(interp_quat)
        else: # Should not happen
            interp_pos, interp_quat = start_p, start_q
        return interp_pos, interp_quat

    def _step_and_record_frame(self) -> bool: # Renamed
        """Steps the simulation and writes one frame via the recorder."""
        if not self.env or not self.camera: return False # Recorder checked separately

        # Step Environment
        dummy_action = self.env.action_space.sample()
        self.env.step(dummy_action)
        self.total_steps_taken += 1 # Increment steps regardless of write success

        # Check if recorder is active before getting obs/processing
        if not self.recorder.is_open():
            # Allow simulation to continue even if not recording
            print("Debug: Recorder not open, skipping frame write.") # Optional debug msg
            return True

        # Get Observation and Write Frame
        try:
            viewer_obs = self.camera.get_obs()
            if not viewer_obs or not isinstance(viewer_obs[0], dict) or 'rgb' not in viewer_obs[0]:
                 print("Warning: Could not get RGB frame from camera observation.")
                 return True # Continue sim, skip frame

            rgb_frame_tensor = viewer_obs[0]['rgb']
            bgr_frame_uint8 = self._process_frame_for_video(rgb_frame_tensor)
            if bgr_frame_uint8 is None: return True # Skip frame if processing failed

            # Tell the recorder instance to write the frame
            self.recorder.write_frame(bgr_frame_uint8)
            return True
        except Exception as e:
            print(f"Error processing/recording frame at step {self.total_steps_taken}: {e}")
            return False # Treat frame error as critical for now

    def _process_frame_for_video(self, rgb_tensor: th.Tensor) -> Optional[NumpyArray]:
        """Converts RGB tensor observation to BGR uint8 numpy array."""
        rgb_frame = rgb_tensor.cpu().numpy()
        if rgb_frame.shape[2] == 4: # RGBA
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGBA2BGR)
        elif rgb_frame.shape[2] == 3: # RGB
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        else:
            print("Warning: Frame is not RGB or RGBA.")
            return None

        if bgr_frame.dtype != np.uint8:
            # Assuming float 0-1 range from OmniGibson
            bgr_frame_uint8 = (np.clip(bgr_frame, 0, 1) * 255).astype(np.uint8)
        else:
            bgr_frame_uint8 = bgr_frame
        return bgr_frame_uint8

    # --- Cleanup ---

    def cleanup(self):
        """Releases resources and performs a full clear of the simulation for the current scene controller."""
        print(f"*"*20 + f" CLEANUP Controller for scene: {self.config.scene_model} " + "*"*20)
        
        # Always try to release the recorder
        self.recorder.release()
        print(f"  Recorder released for {self.config.scene_model}.")

        if self.env is not None:
            print(f"  Attempting to clear simulation state for {self.config.scene_model} using og.clear()...")
            try:
                og.clear() # This will stop, close stage, clear sim instance, and re-launch a new sim
                print(f"  og.clear() successfully called after processing scene {self.config.scene_model}.")
            except Exception as e:
                print(f"  Error during og.clear() for scene {self.config.scene_model}: {e}")
                # Even if og.clear() fails, the old self.env is likely invalid or tied to a problematic state.
            
            self.env = None # The old environment instance is no longer valid.
            print(f"  self.env set to None for {self.config.scene_model}.")
        else:
            print(f"  No active self.env for this controller (scene: {self.config.scene_model}), cleanup primarily for recorder.")

        # The global og.shutdown() is handled at the very end of the main script (generate_data.py).

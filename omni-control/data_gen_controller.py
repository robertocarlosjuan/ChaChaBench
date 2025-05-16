# omnig-control/data_generator.py
import omnigibson as og
import cv2
import os
import numpy as np
import torch as th
import math
import traceback
from typing import List, Tuple, Optional, Any, Dict

# Use relative imports
from .config import SceneConfiguration
from .controller import SceneCameraController, PosOri, NumpyArray # Import types from controller
from .utils import SPACE_CHECK_AMOUNT, CameraMotion, MotionType
# Recorder class is used via inheritance

class DataGeneratorController(SceneCameraController):
    """
    Extends SceneCameraController to generate video data for basic commands.
    It uses the parent's setup() to find a valid starting pose (including path pre-check
    based on config.commands) and then pre-checks and records individual basic commands
    from that pose.
    """
    def __init__(self, config: SceneConfiguration, base_output_dir: str):
        super().__init__(config)
        self.base_output_dir = base_output_dir
        # self.recorder is inherited and initialized in parent __init__
        # Note: self.initial_pos and self.initial_ori will be set by parent's setup()

    def setup_for_data_generation(self) -> bool:
        """
        Performs setup using the parent class's setup method.
        This finds a valid initial pose that passes the pre-check for the
        commands defined in the config file.
        """
        print(f"Setting up scene {self.config.scene_model} for data generation using parent setup...")
        # Parent setup no longer handles recorder init
        setup_ok = super().setup()
        if setup_ok:
             print(f"Parent setup successful.")
            #  if self.eroded_map is None:
            #      print("Error: Eroded map not generated during parent setup.")
            #      return False
             return True
        else:
             print("Parent setup failed or did not yield a valid initial pose.")
             return False
        
    def _get_command_sequence_with_surrounding_space(self) -> List[Tuple[str, str, float]]:
        """
        Returns a command sequence with some surrounding space.
        """
        translational_command_sequence = []
        translational_cmds = [CameraMotion.FORWARD, CameraMotion.BACKWARD, CameraMotion.LEFT, CameraMotion.RIGHT, CameraMotion.UP, CameraMotion.DOWN]
        for cmd in translational_cmds:
            translational_command_sequence.append((MotionType.TRANSLATIONAL, cmd, SPACE_CHECK_AMOUNT))
            translational_command_sequence.append((MotionType.TRANSLATIONAL, cmd, -SPACE_CHECK_AMOUNT))

        return translational_command_sequence

    def _check_command_sequence_validity(self, valid_path_found: bool) -> bool:
        """
        Checks if the command sequence is valid.
        """
        if not valid_path_found:
            print("Error: Failed to find valid path for command sequence.")
            return None
        if self.initial_pos is None or self.initial_ori is None:
            print("Error: Initial pose not set.")
            return None
        if self.eroded_map is None or self.trav_map_obj is None: # Check needed resources
            print("Error: Map objects not ready for pre-check.")
            return None 
        
    def record_command_sequence(self, command_sequence: List[Tuple[str, str, float]], base_filename: str, ensure_some_surrounding_space: bool=False) -> Optional[Dict[str, Any]]:
        """
        Pre-checks, records a video for a single command, returns metadata.
        """
        if ensure_some_surrounding_space:
            """
            Done to ensure that start position has some surrounding space, so there are more cues for VLMs. 
            Makes samples easier to solve.
            Otherwise, start point is close to the object and it's hard for VLMs to predict the motion.
            """
            surroundng_check_command_sequence = self._get_command_sequence_with_surrounding_space()
            valid_path_found = self._find_valid_start_pose_with_precheck(surroundng_check_command_sequence+command_sequence)
            self._check_command_sequence_validity(valid_path_found)
        else:
            valid_path_found = self._find_valid_start_pose_with_precheck(command_sequence)
            self._check_command_sequence_validity(valid_path_found)
        

        # --- 2. Prepare Recording (using parent recorder instance) ---
        relative_video_path, final_video_path = self._get_unique_video_path(base_filename)
        if final_video_path is None: return None

        if not self._prepare_for_command_recording(final_video_path):
            return None

        # --- 3. Execute Simulation ---
        command_success, duration2pose = self._execute_and_get_end_pose(command_sequence)

        # --- 4. Finalize ---
        metadata = self._finalize_command_recording(
            command_sequence=command_sequence,
            command_success=command_success,
            duration2pose=duration2pose,
            relative_video_path=relative_video_path,
            final_video_path=final_video_path
        )

        return metadata

    # --- Helper Methods for Single Command Recording ---

    def _get_unique_video_path(self, base_filename: str) -> Tuple[Optional[str], Optional[str]]:
        """Determines a unique video path within the scene's subdirectory."""
        try:
            scene_video_dir = os.path.join(self.base_output_dir, self.config.scene_model)
            os.makedirs(scene_video_dir, exist_ok=True)
            video_idx = 0
            final_video_path = os.path.join(scene_video_dir, f"{base_filename}_{video_idx}.mp4")
            # Use dirname of base_output_dir to make relative path start from 'videos/' or similar
            output_root_dir = os.path.dirname(self.base_output_dir) if os.path.dirname(self.base_output_dir) else '.'
            relative_video_path = os.path.relpath(final_video_path, output_root_dir)

            while os.path.exists(final_video_path): # Check for existing file
                # print(f"    Path exists: {final_video_path}. Incrementing index.") # Can be verbose
                video_idx += 1
                final_video_path = os.path.join(scene_video_dir, f"{base_filename}_{video_idx}.mp4")
                relative_video_path = os.path.relpath(final_video_path, output_root_dir)
            return relative_video_path, final_video_path
        except Exception as e:
            print(f"Error determining video path for {base_filename}: {e}")
            return None, None

    def _prepare_for_command_recording(self, video_path: str) -> bool:
        """Starts the recorder and resets controller state for a command."""
        # 1. Start the recorder instance for the specific path
        record_started = self.recorder.start(
            output_path=video_path,
            fps=self.config.sim_fps,
            frame_width=self.camera.image_width,
            frame_height=self.camera.image_height
        )
        if not record_started:
            return False

        # 2. Reset step counter
        self.total_steps_taken = 0

        # 3. Reset camera pose
        if self.initial_pos is None or self.initial_ori is None: return False
        # self.camera.set_position_orientation(self.initial_pos, self.initial_ori)
        # try: og.sim.step()
        # except Exception as e: print(f"    Warning: Sim step error after pose reset: {e}")
        return True

    def _execute_and_get_end_pose(self, command_sequence: List[Tuple[str, str, float]]) -> Tuple[bool, Optional[PosOri]]:
        """Executes the command in sim and returns success status and end pose."""
        command_success = False
        end_pos, end_ori = None, None

        total_duration2pose = []
        curr_duration = 0.0
        
        try:
            for command in command_sequence:
                command_success, duration2pose = self._execute_single_command_sim(command, curr_duration)
                if command_success:
                    total_duration2pose.extend(duration2pose)
                    curr_duration = duration2pose[-1]['duration']
                else: return False, None
        except Exception as e:
            print(f"    Exception during command execution for {command}: {e}")
            traceback.print_exc()
            command_success = False
        
        return command_success, total_duration2pose
        


    def _finalize_command_recording(self, command_sequence: List[Tuple[str, str, float]], command_success: bool,
                                     duration2pose: List[Dict[str, Any]], relative_video_path: str,
                                     final_video_path: str) -> Optional[Dict[str, Any]]:
        """Releases writer, calculates duration, cleans up failed videos, returns metadata."""
        # duration_sec = self.total_steps_taken / self.config.sim_fps if command_success else 0.0
        duration_sec = self.recorder.get_duration()
        self.recorder.release() # <-- Use the inherited recorder's release method

        if command_success and self.initial_pos is not None and self.initial_ori is not None:
            metadata = {
                "scene_name": self.config.scene_model,
                "video_path": relative_video_path,
                "start_pos": self.initial_pos.tolist(),
                "start_orientation": self.initial_ori.tolist(),
                "command_sequence": [{"type": command[0], "direction_or_axis": command[1], "amount": command[2]} for command in command_sequence],
                "duration2pose": duration2pose,
                "duration_sec": duration_sec
            }
            print(f"    Success. Recorded: {final_video_path}")
            return metadata
        else:
            print(f"    Command failed. Video might be incomplete: {final_video_path}")
            if os.path.exists(final_video_path):
                try: os.remove(final_video_path); print(f"    Removed potentially failed video.")
                except OSError as e: print(f"    Error removing video file {final_video_path}: {e}")
            return None

    # Parent cleanup now handles recorder release
    # def cleanup(self):
    #     super().cleanup()

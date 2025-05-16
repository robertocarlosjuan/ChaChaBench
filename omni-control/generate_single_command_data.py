# omnig-control/generate_data.py

import omnigibson as og
from omnigibson.utils.asset_utils import get_available_og_scenes

import os
import json
import math
import cv2
import torch as th
import random
import traceback
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import itertools # For generating combinations
from collections import defaultdict

# Use relative imports
from .config import SceneConfiguration
from .data_gen_controller import DataGeneratorController # Import the new controller
from .utils import BASIC_COMMANDS, COMMAND_SHORTCUTS
from .regenerate_recordings_ids import scene2command_map

SEED = 42
random.seed(SEED)

# --- Configuration for Data Generation ---
OUTPUT_DIR = "output" # Base directory for outputs relative to where script is run
VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "single_command_videos_regenerated_round3") # Passed to controller
ANNOTATIONS_FILE = os.path.join(OUTPUT_DIR, "single_command_annotations_regenerated_round3.json")

# --- NEW: Helper to format amount for filename ---
def format_amount_for_filename(amount: float) -> str:
    """Formats a float amount into a filename-safe string."""
    # Add 'n' prefix for negative, replace '.' with 'p'
    sign = "n" if amount < 0 else ""
    # Use f-string formatting for controlled precision, e.g., 2 decimal places
    amount_str = f"{abs(amount):.2f}".replace('.', 'p')
    # Example: 0.5 -> 0p50, -1.0 -> n1p00, math.pi -> 3p14
    return f"{sign}{amount_str}"

# --- JSON Handling ---

def load_annotations(filepath: str) -> List[Dict[str, Any]]:
    """Loads annotations from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list): return data
                else: print(f"Warning: File {filepath} not a list."); return []
        except Exception as e:
            print(f"Error loading {filepath}: {e}"); return []
    return []

def save_annotations(filepath: str, data: List[Dict[str, Any]]):
    """Saves annotations to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving annotations to {filepath}: {e}")

# --- Configuration for command generation ---
# MIN_COMMANDS_IN_SEQUENCE = 2
# MAX_COMMANDS_IN_SEQUENCE = 10
# NUM_VALID_SEQUENCES_TO_SAMPLE_PER_LENGTH = 100 # As per your initial goal (user snippet had 2)
MAX_SIMULATION_ATTEMPTS_PER_BASE_SEQUENCE = 10 # Tries for different random amounts

#  --- Helper function to generate all unique base command combinations (tuples of commands) ---
# --- This should be called ONCE, outside the scene loop ---
# def get_all_base_command_sequences(basic_commands, min_len, max_len):
#     all_sequences = defaultdict(list)
#     for k in range(min_len, max_len + 1):
#         seq_list = []
#         for combo_tuple in itertools.combinations(basic_commands, k):
#             # Each combo_tuple is a unique set of commands.
#             # To get a random permutation of this combination as the "base sequence":
#             combo_list = list(combo_tuple)
#             random.shuffle(combo_list)
#             seq_list.append(tuple(combo_list))
#         random.shuffle(seq_list)
#         all_sequences[k] = seq_list
#     return all_sequences

 # --- Helper to randomize amounts in a sequence ---
def sample_movement_amount_for_sequence(command_sequence: List[tuple[str, str]]) -> List[tuple[str, str, float]]:
    cmd_sequence = []
    for cmd in command_sequence:
        cmd_type, cmd_target = cmd # Original amount is placeholder
        if cmd_type == 'move':
            random_amnt = random.uniform(0.5, 1.5)
            cmd_sequence.append((cmd_type, cmd_target, random_amnt))
        elif cmd_type == 'turn':
            random_amnt = random.uniform(math.pi / 4, math.pi)
            cmd_sequence.append((cmd_type, cmd_target, random_amnt))
        else: # Should not happen with current BASIC_COMMANDS
            cmd_sequence.append((cmd_type, cmd_target, 0.0)) # Default if unknown
    return cmd_sequence


# --- Helper to create a unique filename for a sequence ---
def create_sequence_filename_prefix(scene_name_str: str, sequence_cmds: list) -> str:
    parts = [scene_name_str]
    for cmd_type, direction_or_axis, amount in sequence_cmds:
        type_shortcut = cmd_type[0].upper()
        dir_shortcut = COMMAND_SHORTCUTS.get(cmd_type, {}).get(direction_or_axis, "Unk")
        amount_str = format_amount_for_filename(amount)
        parts.append(f"{type_shortcut}{dir_shortcut}{amount_str}")
    return "_".join(parts)

# --- Main Data Generation Logic ---

def generate_data():
    """Generates video data and annotations across multiple scenes."""
    print("--- Starting Data Generation ---")
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True) # Ensure base video dir exists

    annotations_list = load_annotations(ANNOTATIONS_FILE)
    print(f"Loaded {len(annotations_list)} existing annotation entries.")
    # Note: Duplicate checking logic removed for simplicity, assuming clean runs or manual checks needed.

    try:
        available_scenes = get_available_og_scenes()
        if not available_scenes: print("Error: No available scenes found."); return
        print(f"Found {len(available_scenes)} available scenes.")
    except Exception as e:
        print(f"Error getting available scenes: {e}"); return

    # Main loop through scenes - WRAPPED WITH TQDM
    print(f"\n\navailable_scenes: {available_scenes}\n\n")
    # commands_matched = 0
    basic_commands = BASIC_COMMANDS
    for scene_index, scene_name in enumerate(tqdm(available_scenes, desc="Processing Scenes")):
        # if scene_name not in scene2command_map:
        #     continue
        # basic_commands = scene2command_map[scene_name]
        # commands_matched+=len(basic_commands)
        print(f"\n--- Processing Scene {scene_index + 1}/{len(available_scenes)}: {scene_name} ---")

        config = SceneConfiguration(scene_model=scene_name)
        print(f"updated custom_map_path: {config.custom_map_path}")
        # config.scene_model = scene_name # Set current scene

        # Instantiate the data generator controller
        controller = DataGeneratorController(config, VIDEO_OUTPUT_DIR)
        setup_success = False
        controller_instance = None # Keep reference for cleanup

        try:
            controller_instance = controller # Store ref for finally block
            if controller.setup_for_data_generation():
                setup_success = True
            else:
                 print(f"  Setup failed for scene {scene_name}. Skipping.")

            if setup_success:
                # Loop through basic commands for this scene
                #save all_command_combinations to file
                
                for command in tqdm(basic_commands, desc="Executing Commands"):
                    command_sequence = [command]
                    movement_amount_tried = 0
                    while movement_amount_tried<config.max_movement_amount_tries:
                        command_sequence_randomized = sample_movement_amount_for_sequence(command_sequence)

                        print(f"\n\ncommand_sequence_randomized: {command_sequence_randomized}\n\n")

                        base_filename = create_sequence_filename_prefix(scene_name, command_sequence_randomized)
                        # --- End UPDATED Filename Generation ---
                        print(f"\n\nbase_filename: {base_filename}\n\n")

                        # Record the command and get metadata
                        metadata = controller.record_command_sequence(command_sequence_randomized, base_filename)
                        
                        if metadata:
                            break
                        else:
                            movement_amount_tried+=1
                        

                    if metadata:
                        annotations_list.append(metadata)
                        save_annotations(ANNOTATIONS_FILE, annotations_list) # Save after each success
                    else:
                        print(f"  Failed to record command sequence for {command_sequence}. Skipping.")
                        # unfit_command_combinations.append(command_sequence)
                    # Error messages printed within record_single_command if it fails
                    
                    # all_command_combinations[combo_idx] = unfit_command_combinations+curr_command_combinations 

        except Exception as e:
            print(f"  --- Unhandled Exception during processing scene {scene_name}: ---")
            print(e)
            traceback.print_exc()
            print("  -------------------------------------------------------------")
        finally:
            # Ensure cleanup for the current scene controller instance
            if controller_instance is not None:
                 print(f"  Cleaning up controller for scene {scene_name}...")
                 controller_instance.cleanup()
            print(f"  Finished processing scene {scene_name}.")
    # print(f"commands_matched = {commands_matched}")
    # If a global shutdown is desired after all scenes, do it here:
    if og.sim is not None: # Check if a simulation instance might still exist globally
        print("Shutting down global OmniGibson simulation context...")
        og.shutdown()

    print("\n--- Data Generation Finished ---")
    print(f"Total annotations saved: {len(annotations_list)}")

if __name__ == "__main__":
    generate_data()
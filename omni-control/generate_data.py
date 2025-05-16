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

SEED = 42
random.seed(SEED)

# --- Configuration for Data Generation ---
OUTPUT_DIR = "output" # Base directory for outputs relative to where script is run
VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "combo_videos") # Passed to controller
ANNOTATIONS_FILE = os.path.join(OUTPUT_DIR, "combo_annotations.json")
RESUME_DATA_GENERATION = True
PAST_GENERATED_DATA_FILE = os.path.join(OUTPUT_DIR, "combo_annotations.json")

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
MIN_COMMANDS_IN_SEQUENCE = 2
MAX_COMMANDS_IN_SEQUENCE = 10
NUM_VALID_SEQUENCES_TO_SAMPLE_PER_LENGTH = 100 # As per your initial goal (user snippet had 2)
MAX_SIMULATION_ATTEMPTS_PER_BASE_SEQUENCE = 10 # Tries for different random amounts

#  --- Helper function to generate all unique base command combinations (tuples of commands) ---
# --- This should be called ONCE, outside the scene loop ---
def get_all_base_command_sequences(basic_commands, min_len, max_len):
    all_sequences = defaultdict(list)
    for k in range(min_len, max_len + 1):
        seq_list = []
        for combo_tuple in itertools.combinations(basic_commands, k):
            # Each combo_tuple is a unique set of commands.
            # To get a random permutation of this combination as the "base sequence":
            combo_list = list(combo_tuple)
            random.shuffle(combo_list)
            seq_list.append(tuple(combo_list))
        random.shuffle(seq_list)
        all_sequences[k] = seq_list
    return all_sequences

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

def get_past_generated_data_sequences(past_generated_data_file):
    with open(past_generated_data_file, "r") as f:
        combo_annotations = json.load(f)

    done_sequences = defaultdict(list)
    for annotation in combo_annotations:
        command_sequence = tuple((command["type"], command["direction_or_axis"]) for command in annotation["command_sequence"])
        done_sequences[len(annotation["command_sequence"])].append(command_sequence)

    for key in done_sequences.keys():
        done_sequences[key] = done_sequences[key][::-1]

    return done_sequences

def resume_data_generation(all_sequences):
    done_sequences = get_past_generated_data_sequences(PAST_GENERATED_DATA_FILE)
    
    for combo_idx, all_cmd_seqes in all_sequences.items():
        left_over_sequences = []
        # print(cmd_seqes)

        done_cmd_seq = done_sequences[combo_idx].pop()

        while len(done_sequences[combo_idx]) > 0:
            all_cmd_seq = all_cmd_seqes.pop()
            # print(all_cmd_seq, done_cmd_seq)
            
            if done_cmd_seq == all_cmd_seq:
                done_cmd_seq = done_sequences[combo_idx].pop()
                # exit()
                continue
            else:
                left_over_sequences.append(all_cmd_seq)
    
    all_sequences[combo_idx] = left_over_sequences + all_sequences[combo_idx]
    print(f"Left over sequences: {len(left_over_sequences)}, done sequences: {len(done_sequences[combo_idx])}, total sequences: {len(all_sequences[combo_idx])}")
    return all_sequences



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
    

    all_command_combinations = get_all_base_command_sequences(BASIC_COMMANDS, MIN_COMMANDS_IN_SEQUENCE, MAX_COMMANDS_IN_SEQUENCE)
    if RESUME_DATA_GENERATION:
        all_command_combinations = resume_data_generation(all_command_combinations)
    print(f"Generated {len(all_command_combinations)} unique base command sequences.")
    with open(os.path.join(OUTPUT_DIR, "all_command_combinations.json"), "w") as f:
        json.dump(all_command_combinations, f, indent=4)
    print(f"Lengths of command combinations: {[f'{i}: {len(all_command_combinations[i])}' for i in all_command_combinations.keys()]}")
    # exit()

    # Main loop through scenes - WRAPPED WITH TQDM
    for scene_index, scene_name in enumerate(tqdm(available_scenes[20:], desc="Processing Scenes")):
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
                
                for combo_idx in tqdm(all_command_combinations.keys(), desc="Executing Commands"):
                    curr_command_combinations = all_command_combinations[combo_idx]
                    unfit_command_combinations = []
                    num_cmds_done=0
                    num_cmds_tried=0
                    metadata=None
                    while curr_command_combinations != [] and num_cmds_done<2:
                        command_sequence = curr_command_combinations.pop()
                        print(f"  Executing command {num_cmds_tried + 1}/{len(curr_command_combinations)}: {command_sequence}")
                        num_cmds_tried+=1
                        movement_amount_tried=0

                        while movement_amount_tried<config.max_movement_amount_tries:
                            command_sequence_randomized = sample_movement_amount_for_sequence(command_sequence)

                            base_filename = create_sequence_filename_prefix(scene_name, command_sequence_randomized)
                            # --- End UPDATED Filename Generation ---

                            # Record the command and get metadata
                            metadata = controller.record_command_sequence(command_sequence_randomized, base_filename)
                            
                            if metadata:
                                break
                            else:
                                movement_amount_tried+=1
                            

                        if metadata:
                            annotations_list.append(metadata)
                            save_annotations(ANNOTATIONS_FILE, annotations_list) # Save after each success
                            num_cmds_done+=1
                        else:
                            unfit_command_combinations.append(command_sequence)
                        # Error messages printed within record_single_command if it fails
                    
                    all_command_combinations[combo_idx] = unfit_command_combinations+curr_command_combinations 

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

    # If a global shutdown is desired after all scenes, do it here:
    if og.sim is not None: # Check if a simulation instance might still exist globally
        print("Shutting down global OmniGibson simulation context...")
        og.shutdown()

    print("\n--- Data Generation Finished ---")
    print(f"Total annotations saved: {len(annotations_list)}")

if __name__ == "__main__":
    generate_data()
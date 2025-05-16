import json
import os
import shutil

from tqdm import tqdm

def merge_and_process_annotations(annotations_file_path, regenerated_annotations_file_path, output_video_dir):
    """
    Merges two annotation files, performs sanity checks, and copies relevant videos.

    Args:
        annotations_file_path (str): Path to the original annotations JSON file.
        regenerated_annotations_file_path (str): Path to the regenerated annotations JSON file.
        output_video_dir (str): Path to the directory where videos will be copied.
    """
    try:
        with open(annotations_file_path, 'r') as f:
            annotations_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {annotations_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from - {annotations_file_path}")
        return

    try:
        with open(regenerated_annotations_file_path, 'r') as f:
            regenerated_annotations_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found - {regenerated_annotations_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from - {regenerated_annotations_file_path}")
        return

    updated_samples_count = 0
    updated_commands = set()
    processed_regenerated_indices = set() # To avoid reusing regenerated samples

    # Create a lookup for regenerated annotations for faster matching
    regenerated_lookup = {}
    for i, regen_sample in enumerate(regenerated_annotations_data):
        if regen_sample and isinstance(regen_sample.get("command_sequence")[0], dict):
            key = (regen_sample.get("scene_name"), regen_sample["command_sequence"][0].get("direction_or_axis"))
            if key not in regenerated_lookup: # Keep the first encountered sample if duplicates exist
                regenerated_lookup[key] = (regen_sample, i)
            else:
                print(f"Strange: Duplicate regenerated sample found for key: {key}")


    merged_annotations = []
    updated_scenes = set()
    for original_sample in annotations_data:
        if not original_sample or not isinstance(original_sample.get("command_sequence")[0], dict):
            print(f"Strange: Original sample command sequence is not a dictionary: {original_sample}/n appending to merged_annotations for now.")
            merged_annotations.append(original_sample)
            continue

        original_scene_name = original_sample.get("scene_name")
        original_direction = original_sample["command_sequence"][0].get("direction_or_axis")
        
        match_key = (original_scene_name, original_direction)

        if match_key in regenerated_lookup:
            regenerated_sample, regen_idx = regenerated_lookup[match_key]
            if regen_idx not in processed_regenerated_indices:
                # Update original_sample with regenerated_sample values
                # This creates a new dictionary to avoid modifying the original_sample in-place
                # before it's decided if it's truly the one to update (in case of multiple matches in source)
                updated_sample_copy = original_sample.copy()
                updated_sample_copy.update(regenerated_sample)
                merged_annotations.append(updated_sample_copy)
                
                updated_samples_count += 1
                updated_scenes.add(original_scene_name)
                if original_direction: # Ensure direction is not None
                    updated_commands.add(original_direction)
                processed_regenerated_indices.add(regen_idx)
                # Remove from lookup to ensure one-to-one mapping if multiple original samples could match one regenerated
                del regenerated_lookup[match_key] 
            else:
                # This regenerated sample was already used for a previous match
                print(f"Strange: Regenerated sample {regen_idx} was already used for a previous match")
                merged_annotations.append(original_sample)
        else:
            merged_annotations.append(original_sample)

    print(f"Number of samples updated: {updated_samples_count}")
    print(f"Number scenes updated: {len(updated_scenes)}")
    print(f"Number of unique commands updated: {len(updated_commands)}")
    print(f"Updated commands: {sorted(list(updated_commands))}")
    print(f"Final number of samples: {len(merged_annotations)}")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
        print(f"Created directory: {output_video_dir}")
    else:
        print(f"Directory already exists: {output_video_dir}")

    copied_videos_count = 0
    for sample in tqdm(merged_annotations):
        if sample and "video_path" in sample and sample["video_path"]:
            original_video_path = sample["video_path"]
            # if not os.path.isabs(original_video_path):
            #     print(f"Strange: Video path is not absolute: {original_video_path}")

            video_filename = os.path.basename(original_video_path)
            destination_video_path = output_video_dir + '/' + sample.get("scene_name") + '/' + video_filename
            # make the directory if it doesn't exist
            os.makedirs(os.path.dirname(destination_video_path), exist_ok=True)

            full_path = base_dir + '/' + original_video_path
            try:
                if os.path.exists(full_path):
                    shutil.copy(full_path, destination_video_path)

                    rel_path = output_video_dir.split('/')[-1] + '/' + sample.get("scene_name") + '/' + video_filename

                    #update the relative path in merged_annotations
                    sample["video_path"] = rel_path


                    # print(f"Copied: {original_video_path} to {destination_video_path}")
                    copied_videos_count +=1
                else:
                    print(f"Warning: Source video not found, skipping: {original_video_path}")
            except Exception as e:
                print(f"Error copying video {original_video_path} to {destination_video_path}: {e}")
        elif sample and ("video_path" not in sample or not sample["video_path"]):
            print(f"Warning: Sample missing 'video_path' or 'video_path' is empty. Scene: {sample.get('scene_name')}, Command: {sample.get('command_sequence', {}).get('direction_or_axis')}")


    print(f"Number of videos copied: {copied_videos_count}")
    print(f"Total samples in merged data: {len(merged_annotations)}")


    # Save the merged annotations
    
    try:
        with open(merged_annotations_output_path, 'w') as f:
            json.dump(merged_annotations, f, indent=4)
        print(f"Merged annotations saved to: {merged_annotations_output_path}")
    except IOError:
        print(f"Error: Could not write merged annotations to - {merged_annotations_output_path}")

def check_merge_annotations(merged_annotations_output_path):
    with open(merged_annotations_output_path, 'r') as f:
        merged_annotations = json.load(f)
    error_count = 0
    for sample in merged_annotations:
        if sample and "video_path" in sample and sample["video_path"]:
            #check if the video path exists
            if not os.path.exists(base_dir + '/' + sample["video_path"]):
                print(f"Warning: Video path does not exist: {sample['video_path']}")
                error_count += 1
    print(f"Total errors: {error_count}")
    if error_count == 0:
        print("No errors found")
    else:
        print("Errors found")

if __name__ == "__main__":
    # annotations_file = "/nethome/che321/flash/camera-motion/output/single_command_annotations.json"
    annotations_file = "/nethome/che321/flash/camera-motion/output/single_command_annotations_merged_final_round2.json"
    regenerated_annotations_file = "/nethome/che321/flash/camera-motion/output/single_command_annotations_regenerated_round3.json"
    final_videos_dir = "/nethome/che321/flash/camera-motion/output/single_command_videos_final_round3"
    base_dir = "/nethome/che321/flash/camera-motion/output"
    merged_annotations_output_path = '/nethome/che321/flash/camera-motion/output/single_command_annotations_merged_final_round3.json'
    merge_and_process_annotations(annotations_file, regenerated_annotations_file, final_videos_dir)
    check_merge_annotations(merged_annotations_output_path)

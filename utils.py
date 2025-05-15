import re
import json
import time
import google.generativeai as genai

def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data

def save_json(json_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

def get_video_input(video_file_name):
    try:
        video_file = genai.upload_file(path=video_file_name)
    except Exception as e:
        print(f"File upload error: {e}")
        return None

    try:
        while video_file.state.name == "PROCESSING":
            time.sleep(12)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)
    except Exception as e:
        print(f"Error during file processing: {e}")
        return None
    return video_file
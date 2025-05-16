import os
import json
from models.gpt import OpenAIModel
from tqdm import tqdm
from utils import load_json
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'results_cot')
VERIFICATION_DIR = os.path.join(os.path.dirname(__file__), 'verification/reasoning')
MODEL_PATH = 'gpt-4o'  # Change as needed
SELECTED_FILES = [
    "singlecot_gemini-2.5-flash-preview-04-17.json", 
    "singlecot_gemini-2.0-flash.json",
    "singlecot_Qwen2.5-VL-72B-Instruct.json",
    "singlecot_Qwen2.5-VL-7B-Instruct.json",
    "singlecot_qwen2-7b-longvila-256f.json",
    "singlecot_qwen2.5-vl-7b-cam-motion-preview.json",
    "singlecot_NVILA-15B.json",
    "singlecot_Qwen2.5-VL-32B-Instruct.json"
]

os.makedirs(VERIFICATION_DIR, exist_ok=True)

def evaluate_all_results():
    model = OpenAIModel(MODEL_PATH)
    for filename in SELECTED_FILES:
        if not filename.endswith('.json'):
            continue
        input_path = os.path.join(RESULTS_DIR, filename)
        output_path = os.path.join(VERIFICATION_DIR, filename)
        print(f"Processing {filename}...")
        with open(input_path, 'r') as infile:
            try:
                data = json.load(infile)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                continue
        results = load_json(output_path) if os.path.exists(output_path) else {}
        for video_id, entry in tqdm(data.items()):
            if video_id in results:
                continue
            freeform = entry.get('freeform')
            if not freeform:
                results[video_id] = {'error': 'No freeform attribute'}
                continue
            evaluation = model.evaluate_reasoning(freeform)
            results[video_id] = {
                'freeform': freeform, 
                'evaluation': evaluation, 
                'prediction': entry['structured'], 
                'ground_truth': entry['ground_truth'][0]['camera_motion']
            }
            with open(output_path, 'w') as outfile:
                json.dump(results, outfile)
        print(f"Saved verification to {output_path}")

if __name__ == '__main__':
    evaluate_all_results()






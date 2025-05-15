import os
import argparse
from tqdm import tqdm
from ground_truth_loader import GroundTruthLoader

from utils import load_json, save_json

MODEL_CHOICES = [
        "gemini-2.0-flash", 
        "gemini-2.5-flash-preview-04-17", 
        "Efficient-Large-Model/qwen2-7b-longvila-256f",
        "Efficient-Large-Model/NVILA-15B",
        "chancharikm/qwen2.5-vl-7b-cam-motion-preview",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct"]

def infer_model_class(model_path: str):
    lower_path = model_path.lower()
    if 'vila' in lower_path:
        from models.vila import VILAModel
        return VILAModel, 'vila'
    elif 'qwen' in lower_path:
        from models.qwen import QwenModel
        return QwenModel, 'qwen'
    elif 'gemini' in lower_path:
        from models.gemini import GeminiModel
        return GeminiModel, 'gemini'
    else:
        raise ValueError(f"Could not infer model type from model_path: {model_path}")

def main():
    parser = argparse.ArgumentParser(description="ChaChaBench Model Evaluation (auto model selection)")
    parser.add_argument('--model_path', type=str, required=True, choices=MODEL_CHOICES, help='Model path or identifier (used to infer model type)')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    model_class, model_type = infer_model_class(args.model_path)
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Use model_path as suffix for results file
    safe_model_name = args.model_path.replace('/', '-').replace('.', '-')
    results_file = os.path.join(results_dir, f'{safe_model_name}.json')
    loader = GroundTruthLoader(ground_truth_path=f'data/annotations.json')
    results = load_json(results_file) if os.path.exists(results_file) else {}

    # Model-specific instantiation and evaluation function
    model = model_class(model_path=args.model_path)

    for i in tqdm(range(len(loader))):
        video_id = loader.get_video_id(i)
        if video_id in results:
            continue
        video_path, ground_truth = loader[i]
        try:
            cot_response, answer, structured = model.cot_single_command(video_path)
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue
        results[video_id] = {
            'freeform': cot_response,
            'structured_text': structured,
            'structured': answer,
            'ground_truth': ground_truth,
        }
        save_json(results, results_file)

if __name__ == '__main__':
    main()
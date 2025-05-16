import cv2
import os
import json
import numpy as np
from openai import OpenAI
from prompts.verification_prompts import reasoning_system_prompt, reasoning_prompt, scene_system_prompt, scene_prompt
from dotenv import load_dotenv
import base64
import io
from PIL import Image
load_dotenv()




class OpenAIModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.client = OpenAI()

    def evaluate_reasoning(self, cot_text):
        try:
            response = self.client.responses.create(
                model=self.model_path,
                input = reasoning_prompt.format(cot_text),
                instructions = reasoning_system_prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "camera_motion_evaluation",
                        "schema": {
                            "type": "object",
                            "description": "Evaluation of camera motion reasoning and final answer consistency.",
                            "properties": {
                                "logical_consistency": {
                                    "type": "string",
                                    "enum": ["Yes", "No"],
                                    "description": "Whether the reasoning maintains internal logical consistency."
                                },
                                "consistency_reason": {
                                    "type": "string",
                                    "description": "Brief explanation of why the reasoning is or isn't logically consistent."
                                },
                                "final_answer_match": {
                                    "type": "string",
                                    "enum": ["Yes", "No"],
                                    "description": "Whether the final answer logically follows from the reasoning."
                                },
                                "match_reason": {
                                    "type": "string",
                                    "description": "Brief explanation of why the final answer matches or does not match the reasoning."
                                }
                            },
                            "required": ["logical_consistency", "consistency_reason", "final_answer_match", "match_reason"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            return json.loads(response.output_text)
        except Exception as e:
            print(f"Error getting assessment from ChatGPT: {str(e)}")
            return {}




    def extract_uniform_frames_base64(self, video_path, num_frames=5):
        vidcap = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0 or num_frames == 0:
            vidcap.release()
            return []

        # Compute uniform frame indices
        sample_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

        base64_frames = []
        current_idx = 0

        for target_idx in sample_indices:
            # Efficient seeking
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            success, frame = vidcap.read()

            if not success:
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_img = Image.fromarray(frame_rgb)
            # Encode to base64
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_frames.append(img_base64)

        vidcap.release()
        return base64_frames

    def evaluate_scene_motion_grounding(self, video_path, description):
        base64_images = self.extract_uniform_frames_base64(video_path)
        base64_images_content = [{"type": "input_image", "image_url": f"data:image/png;base64,{frame_b64}"} for frame_b64 in base64_images[1:6]]
        try:
            response = self.client.responses.create(
                model=self.model_path,
                input=[
                    {
                        "role": "system",
                        "content": scene_system_prompt
                    },

                    {
                        "role": "user",
                        "content": scene_prompt.format(description),
                    },
                    {
                        "role": "user",
                        "content": base64_images_content,
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "camera_motion_evaluation",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "grounding_verdict": {
                                    "type": "string",
                                    "enum": ["Grounded", "Not Grounded"],
                                    "description": "Whether the scene description accurately corresponds to the visual content of the image."
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Summary of matches/mismatches between description and image."
                                },
                                "object_mentions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Objects explicitly mentioned in the description."
                                },
                                "missing_objects": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Objects mentioned but not visible in the image."
                                },
                                "incorrect_motion_descriptions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Description fragments where described motion/spatial relation mismatches the image."
                                }
                            },
                            "required": [
                                "grounding_verdict",
                                "reason",
                                "object_mentions",
                                "missing_objects",
                                "incorrect_motion_descriptions"
                            ],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            return json.loads(response.output_text)
        except Exception as e:
            print(f"Error getting assessment from ChatGPT: {str(e)}")
            return {}



import os
import re
# Local application/library specific imports
from utils import get_video_input
from prompts.single_prompts import generation_config, camera_motion_labels, gemini_cot_prompt

import dotenv
import google.generativeai as genai
dotenv.load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class GeminiModel:
    def __init__(self, model_path: str):
        self.model = genai.GenerativeModel(model_name=model_path)
        self.generation_config = structured_generation_config = {
            **generation_config,
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "string",
                "enum": camera_motion_labels + ["None of the above"]
            }
        }

    def cot_single_command(self, video_path):
        video_input = get_video_input(video_path)
        # skip if video input is None
        if video_input is None:
            return None, None, None
        
        try:
            cot_response = self.model.generate_content(
                    [video_input, gemini_cot_prompt],
                    request_options={"timeout": 600},
                    generation_config=self.generation_config
                ).text
            structured = re.sub(r'[^a-z\s]', '', cot_response.lower())
            structured = structured.rsplit('final answer', 1)[-1].strip()
        except Exception as e:
            print(f"Error: {e}")
            return None, None, None
        camera_motion = structured if structured in camera_motion_labels else None
        return cot_response, camera_motion, structured
    
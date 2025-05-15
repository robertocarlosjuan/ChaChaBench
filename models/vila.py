import os
import sys
import re
sys.path.append('/nethome/che321/flash/VILA')
import llava
from llava import conversation as clib
from llava.media import Image, Video


# Local application/library specific imports
from utils import get_video_input, extract_json_from_text
from prompts.single_prompts import (
    qwen_cot_system_prompt as single_cot_system_prompt, 
    qwen_cot_prompt as single_cot_prompt,
    qwen_cot_mapper as cot_mapper
)

class VILAModel:
    def __init__(self, model_path = "Efficient-Large-Model/qwen2-7b-longvila-256f"):
        
        self.model = llava.load(model_path) #, devices=devices
        clib.default_conversation = clib.conv_templates["auto"].copy()
        self.config = self.model.default_generation_config
        self.config.temperature = 0
        self.config.top_p = 1
        self.config.top_k = 1
        self.config.max_new_tokens=512
        self.config.do_sample = False


    def run(self, conversation):
        response = self.model.generate_content_from_conversation(conversation, generation_config=self.config)
        return response

    def cot_single_command(self, video_path):
        media = Video(video_path)
        prompt = [media, single_cot_prompt]
        freeform_conversation = [{"from": "human", "value": prompt}]
        cot_response = self.run(freeform_conversation)
        camera_motion = None
        structured = None
        try:
            structured = re.sub(r'[^a-z\s]', '', cot_response.lower())
            structured = structured.rsplit('final answer', 1)[-1].strip().split(' ')[0].strip().upper()
            if structured not in cot_mapper:
                camera_motion = None
            else:
                camera_motion = cot_mapper[structured]
        except Exception as e:
            print(f"Error: {e}")
            camera_motion = None
        return cot_response, camera_motion, structured
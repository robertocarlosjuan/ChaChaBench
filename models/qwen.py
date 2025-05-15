import os
import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# Local application/library specific imports
from utils import get_video_input, extract_json_from_text

from prompts.single_prompts import (
    qwen_cot_system_prompt as single_cot_system_prompt, 
    qwen_cot_prompt as single_cot_prompt,
    qwen_cot_mapper as cot_mapper
)

class QwenModel:
    def __init__(self, model_path):
        # Get number of available GPUs
        self.num_gpus = torch.cuda.device_count()
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            use_fast=True,
            torch_dtype=torch.bfloat16,
        )

        self.generation_config = generation_config = {
            'temperature': 0,
            'top_p': 1,
            'top_k': 1,
            'do_sample': False
        }

    def run(self, message, generation_config=None):
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(message, return_video_kwargs=True)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=512,
            **generation_config
            )
        
        # Check if generation was successful
        if generated_ids is None or len(generated_ids) == 0:
            print("Warning: Null generation")
            return ["Error: Generation failed"]
            
        prompt_ids = inputs.input_ids[0]
        generated_ids_trimmed = []
        for i, out_ids in enumerate(generated_ids):
            trimmed = out_ids[len(prompt_ids) :]
            generated_ids_trimmed.append(trimmed)

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if output_text and any(text.strip() for text in output_text):
            if type(output_text) == list:
                return output_text[0]
            else:
                return output_text
        else:
            print("Warning: Empty output")
            return ["Error: No valid output generated"]

    def cot_single_command(self, video_path, fps=1.0):
        freeform_message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": single_cot_system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps,
                    },
                    {
                        "type": "text",
                        "text": single_cot_prompt
                    }
                ]
            }
        ]
        cot_response = self.run(freeform_message, generation_config=self.generation_config)
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
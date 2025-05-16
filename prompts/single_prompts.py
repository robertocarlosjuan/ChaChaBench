camera_motion_labels = [
    "move forward",
    "move backward",
    "move left",
    "move right",
    "move up",
    "move down",
    "pan left",
    "pan right",
    "tilt up",
    "tilt down",
    "roll counter-clockwise",
    "roll clockwise",
]

generation_config = {
    "temperature": 0.0,
    "top_p": 1,
    "top_k": 1,
}

gemini_cot_prompt = """
You are a camera-motion expert.
I am going to give you a video and your task is to identify how the camera moves or rotates in the scene from the camera's perspective.

Try to reason about how the camera moves step by step to help you get the correct answer. 
First, please describe how the camera moves in the scene from the camera's perspective. 
Then, respond with your reason first.
Lastly, always finish your answer with ’Final Answer: (X)’, where X is the correct category.
If none or more than one of the options match, respond with "none of the above".

Options:

• pan left: The camera rotates its angle by pivoting left with respect to the camera frame.  
• pan right: The camera rotates its angle by pivoting right with respect to the camera frame.  
• tilt up: The camera rotates its angle up vertically with respect to the initial frame.  
• tilt down: The camera rotates its angle down vertically with respect to the initial frame.  
• roll counter-clockwise: The camera performs a clear and consistent counter-clockwise (CCW) roll by rotating around its own optical center.  
• roll clockwise: The camera performs a clear and consistent clockwise (CW) roll by rotating around its own optical center.
• move forward: The camera physically moves forward relative to the orientation of the camera frame.  
• move backward: The camera physically moves backward relative to the orientation of the camera frame.  
• move left: The camera physically moves left relative to the orientation of the camera frame.  
• move right: The camera physically moves right relative to the orientation of the camera frame.  
• move up: The camera physically moves up relative to the orientation of the camera frame.  
• move down: The camera physically moves down relative to the orientation of the camera frame.
"""

qwen_cot_system_prompt = "You are a camera-motion expert. Please reason step by step, and put your final answer in the format 'Final Answer: <answer>'."

qwen_cot_prompt = """You are a camera-motion expert.
I am going to give you a video and your task is to identify how the camera moves or rotates in the scene from the camera's perspective.
Try to reason about how the camera moves step by step to help you get the correct answer. 

Use the following format:

Thought: reason about how the camera moves step by step to help you get the correct answer
Final Answer: Please select the correct answer from the options below. If none or more than one of the options match, respond with "Final Answer: N".

Options:

A. pan left: The camera rotates its angle by pivoting left with respect to the camera frame.
B. pan right: The camera rotates its angle by pivoting right with respect to the camera frame.
C. tilt up: The camera rotates its angle up vertically with respect to the initial frame.
D. tilt down: The camera rotates its angle down vertically with respect to the initial frame.
E. roll counter-clockwise: The camera performs a clear and consistent counter-clockwise (CCW) roll by rotating around its own optical center.
F. roll clockwise: The camera performs a clear and consistent clockwise (CW) roll by rotating around its own optical center.
G. move forward: The camera physically moves forward relative to the orientation of the camera frame.
H. move backward: The camera physically moves backward relative to the orientation of the camera frame.
I. move left: The camera physically moves left relative to the orientation of the camera frame.
J. move right: The camera physically moves right relative to the orientation of the camera frame.
K. move up: The camera physically moves up relative to the orientation of the camera frame.
L. move down: The camera physically moves down relative to the orientation of the camera frame.
"""

qwen_cot_mapper = {
    "A": "pan left",
    "B": "pan right",
    "C": "tilt up",
    "D": "tilt down",
    "E": "roll counter-clockwise",
    "F": "roll clockwise",
    "G": "move forward",
    "H": "move backward",
    "I": "move left",
    "J": "move right",
    "K": "move up",
    "L": "move down",
    "N": "none of the above",
}
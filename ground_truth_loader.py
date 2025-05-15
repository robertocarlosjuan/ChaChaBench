import json
import os
import random
from utils import load_json

class GroundTruthLoader:
    def __init__(self, root_folder='.', ground_truth_path='data/video_annotations.json'):
        self.root_folder = os.path.abspath(root_folder)
        self.ground_truth_path = os.path.abspath(ground_truth_path)
        self.ground_truth, self.video_paths = self.load()
        self.video_ids = sorted(list(self.video_paths.keys()), key = lambda x: self.get_num_commands(x))

    def __len__(self):
        return len(self.video_ids)

    def randomize_videoids(self):
        random.shuffle(self.video_ids)

    def reverse_videoids(self):
        self.video_ids = list(reversed(self.video_ids))

    def get_num_commands(self, video_id):
        return len(self.ground_truth[video_id])
        
    def get_video_id(self, index):
        return self.video_ids[index]

    def __getitem__(self, index):
        video_id = self.get_video_id(index)
        return self.video_paths[video_id], self.ground_truth[video_id]


    def _get_time_segments(self, duration2pose):
        time_segments = []
        start_time = 0
        for item in duration2pose:
            end_time = item['duration']
            time_segments.append((start_time, end_time))
            start_time = end_time
        return time_segments

    def _map_command(self, command):
        command_map = {
            'pitch': 'tilt',
            'yaw': 'pan',
            'roll': 'roll'
        }
        movement = command['direction_or_axis']
        if command['type'] == 'turn':
            move_type, direction = movement.split('_')
            move_type = command_map[move_type]
            return f'{move_type} {direction}'
        elif command['type'] == 'move':
            return f'move {movement}'
        else:
            raise ValueError(f'Invalid command type: {command["type"]}')

    def _get_camera_motion(self, command_sequence):
        camera_motion = []
        camera_motion_amount = []
        for original_command in command_sequence:
            mapped_command = self._map_command(original_command)
            camera_motion.append(mapped_command)
            camera_motion_amount.append(original_command['amount'])
        return camera_motion, camera_motion_amount

    def _get_motion_sequence(self, ground_truth_item):
        time_segments = self._get_time_segments(ground_truth_item['duration2pose'])
        camera_motion, camera_motion_amount = self._get_camera_motion(ground_truth_item['command_sequence'])
        motion_sequence = []
        for i in range(len(time_segments)):
            start_time, end_time = time_segments[i]
            motion_sequence.append({
                'start_time': start_time,
                'end_time': end_time,
                'camera_motion': camera_motion[i],
                'camera_motion_amount': camera_motion_amount[i]
            })
        return motion_sequence

    def load(self):
        ground_truth_data = load_json(self.ground_truth_path)
        ground_truth = {}
        video_paths = {}
        for ground_truth_item in ground_truth_data:
            video_id = os.path.splitext(os.path.basename(ground_truth_item['video_path']))[0]
            video_path = os.path.join(self.root_folder, 'data', ground_truth_item['video_path'])
            video_paths[video_id] = video_path
            motion_sequence = self._get_motion_sequence(ground_truth_item)
            ground_truth[video_id] = motion_sequence
        return ground_truth, video_paths
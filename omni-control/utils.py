# omnig-control/utils.py
import torch as th
from typing import List
from enum import Enum

# --- Constants ---
# Camera local axes (relative to camera's own frame)
CAM_AXIS_X = th.tensor([1.0, 0.0, 0.0], dtype=th.float32) # Often Camera Right
CAM_AXIS_Y = th.tensor([0.0, 1.0, 0.0], dtype=th.float32) # Often Camera Up
CAM_AXIS_Z = th.tensor([0.0, 0.0, 1.0], dtype=th.float32) # Often Camera Back (-Forward)

# Mappings from string command to camera axis vector
# Adjust these based on your desired camera coordinate frame interpretation

MOVE_DIRECTION_TO_AXIS = {
    "right":    CAM_AXIS_X,
    "left":    -CAM_AXIS_X,
    "up":       CAM_AXIS_Y,
    "down":    -CAM_AXIS_Y,
    "forward": -CAM_AXIS_Z,  # Forward along -Z camera axis
    "backward": CAM_AXIS_Z,
}

TURN_AXIS_TO_AXIS = {
    "roll_right":     -CAM_AXIS_Z,   # Roll around camera X
    "pitch_up":    CAM_AXIS_X,   # Pitch around camera Y
    "yaw_right":      -CAM_AXIS_Y,   # Yaw around camera Z
    "roll_left":     CAM_AXIS_Z,   # Roll around camera X
    "pitch_down":    -CAM_AXIS_X,   # Pitch around camera Y
    "yaw_left":      CAM_AXIS_Y,   # Yaw around camera Z
}

BASIC_COMMANDS = [
    ("move", "forward"),
    ("move", "backward"),
    ("move", "left"),
    ("move", "right"),
    ("move", "up"),
    ("move", "down"),
    ("turn", "roll_right"),
    ("turn", "pitch_up"),
    ("turn", "yaw_right"),
    ("turn", "roll_left"), # Opposite direction
    ("turn", "pitch_down"),
    ("turn", "yaw_left"),
]

# --- NEW: Command Shortcuts for Filenames ---
COMMAND_SHORTCUTS = {
    "move": {
        "forward": "F", "backward": "B", "left": "L", "right": "R", "up": "U", "down": "D"
    },
    "turn": {
        "roll_right": "RR", "pitch_up": "PU", "yaw_right": "YR", "roll_left": "RL", "pitch_down": "PD", "yaw_left": "YL"
    }
}

# --- Quaternion Math Helpers ---

def quat_multiply(q1: th.Tensor, q2: th.Tensor) -> th.Tensor:
    """Multiplies two quaternions (w last convention assumed)."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return th.stack([x, y, z, w], dim=-1)

def quat_from_axis_angle(axis: List[float], angle: float) -> th.Tensor:
    """Creates a quaternion from an axis and angle (w last convention)."""
    axis_tensor = th.tensor(axis, dtype=th.float32)
    norm = th.linalg.norm(axis_tensor)
    if norm < 1e-6:
        # Return identity quaternion for zero rotation
        return th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32)
    axis_tensor = axis_tensor / norm
    half_angle = th.tensor(angle / 2.0)
    sin_half_angle = th.sin(half_angle)
    cos_half_angle = th.cos(half_angle)
    # quat = [axis * sin, cos]
    quat = th.cat([axis_tensor * sin_half_angle, cos_half_angle.unsqueeze(0)])
    # Normalize for safety, although mathematically it should be normalized
    return quat / th.linalg.norm(quat)

def rotate_vector_by_quat(v: th.Tensor, q: th.Tensor) -> th.Tensor:
    """Rotates vector v (3-element tensor) by quaternion q (w last)."""
    # Ensure v is float32 and on same device as q if necessary
    v = v.to(dtype=q.dtype, device=q.device)
    # Create quaternion representation of vector: [v, 0]
    v_quat = th.cat([v, th.tensor([0.0], device=q.device, dtype=q.dtype)])
    # Calculate conjugate: [-q_xyz, q_w]
    q_conj = q * th.tensor([-1.0, -1.0, -1.0, 1.0], device=q.device, dtype=q.dtype)
    # Compute rotated vector: q * v_quat * q_conj
    rotated_v_quat = quat_multiply(quat_multiply(q, v_quat), q_conj)
    # Return the vector part (first 3 elements)
    return rotated_v_quat[..., :3]

SPACE_CHECK_AMOUNT = 5.0

class MotionType(Enum):
    TRANSLATIONAL = "move"
    ROTATIONAL = "turn"

class CameraMotion(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    ROLL_RIGHT = "roll_right"
    ROLL_LEFT = "roll_left"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"

CAMERA_MOTION_TO_STRING = {
    CameraMotion.FORWARD: "move forward",
    CameraMotion.BACKWARD: "move backward",
    CameraMotion.LEFT: "move left",
    CameraMotion.RIGHT: "move right",
    CameraMotion.UP: "move up",
    CameraMotion.DOWN: "move down",
    CameraMotion.ROLL_RIGHT: "roll right",
    CameraMotion.ROLL_LEFT: "roll left",
    CameraMotion.TILT_UP: "tilt up",
    CameraMotion.TILT_DOWN: "tilt down",
    CameraMotion.PAN_LEFT: "pan left",
    CameraMotion.PAN_RIGHT: "pan right",
}

STRING_TO_CAMERA_MOTION = {v: k for k, v in CAMERA_MOTION_TO_STRING.items()}

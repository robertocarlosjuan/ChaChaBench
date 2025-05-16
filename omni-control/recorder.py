# omni-control/recorder.py
import cv2
import numpy as np
import os
from typing import Optional

class VideoRecorder:
    """Handles the creation, writing, and releasing of a video file."""

    def __init__(self):
        self.writer: Optional[cv2.VideoWriter] = None
        self.is_recording: bool = False
        self.output_path: Optional[str] = None
        self.frame_count: int = 0
        self.fps: int = 0 # Added to store FPS

    def start(self, output_path: str, fps: int, frame_width: int, frame_height: int) -> bool:
        """
        Initializes and opens the video writer for a given path.
        Releases any previously open writer.

        Args:
            output_path: The full path to save the video file.
            fps: Frames per second for the video.
            frame_width: Width of the video frames in pixels.
            frame_height: Height of the video frames in pixels.

        Returns:
            True if initialization was successful, False otherwise.
        """
        self.release() # Ensure any existing writer is closed first

        self.output_path = output_path
        self.fps = fps # Store the FPS
        print(f"  Initializing recorder for: {self.output_path}")
        try:
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Common codec
            self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
            if not self.writer.isOpened():
                raise IOError(f"CV2 VideoWriter failed to open for {self.output_path}")
            self.is_recording = True
            self.frame_count = 0
            print(f"  Recording started to {self.output_path}")
            return True
        except Exception as e:
            print(f"    Error initializing VideoWriter: {e}")
            self.writer = None
            self.is_recording = False
            self.output_path = None
            self.frame_count = 0
            self.fps = 0 # Reset FPS on failure
            return False

    def write_frame(self, frame: np.ndarray):
        """
        Writes a frame to the video file if recording is active.

        Args:
            frame: The frame (NumPy array, typically BGR uint8) to write.
        """
        if self.is_recording and self.writer is not None:
            try:
                self.writer.write(frame)
                self.frame_count += 1
            except Exception as e:
                print(f"Error writing frame to {self.output_path}: {e}")
                # Consider stopping recording on write error if needed
                # self.release()
        elif not self.is_recording:
             print("Warning: Tried to write frame, but recorder is not active.")

    def release(self):
        """Releases the video writer and resets the recorder's state."""
        if self.is_recording and self.writer is not None:
            path_str = self.output_path if self.output_path else "unknown path"
            print(f"  Releasing recorder for {path_str}. Frames written: {self.frame_count}")
            try:
                self.writer.release()
            except Exception as e:
                 print(f"Error releasing video writer: {e}")
        # Reset state regardless of whether it was open or not
        self.writer = None
        self.is_recording = False
        self.output_path = None
        self.frame_count = 0
        # Do not reset self.fps here if you want to get duration after release

    def is_open(self) -> bool:
         """Checks if the writer is initialized and recording."""
         return self.is_recording and self.writer is not None 

    def get_duration(self) -> float:
        """Calculates the duration of the recorded video in seconds."""
        if self.fps > 0:
            return self.frame_count / self.fps
        return 0.0 
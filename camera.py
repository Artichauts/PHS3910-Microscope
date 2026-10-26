
from picamera2 import Picamera2
import cv2
from time import sleep
import tifffile
import time
import os
import numpy as np

def capture_one_frame():
    cam = Picamera2()

    cam.configure(
        cam.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
    )

    cam.start()
    sleep(2)

    frame = cam.capture_array()

    if frame is not None:
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if cv2.imwrite("/home/phs3910/test.jpg", frame):
            print("Image saved successfully!")
        else:
            print("Could not save image")
    else:
        print("Could not capture image")

    cam.stop()
    return frame











def capture_multipage_tiff(num_frames=10, filename="output_stack.tiff", interval=0.1):
    """
    Captures multiple frames using Picamera2 and saves them into a 
    single multi-page TIFF file.
    
    Parameters:
    - num_frames (int): Total number of frames to capture.
    - filename (str): Path and name of the output TIFF file.
    - interval (float): Time delay (in seconds) between captures.
    """
    cam = Picamera2()
    
    # Configure for still image capture
    config = cam.create_still_configuration()
    cam.configure(config)
    
    print("Starting camera and warming up...")
    cam.start()
    time.sleep(2)  # Allow auto-exposure and white balance to settle
    
    frames = []
    
    try:
        print(f"Capturing {num_frames} frames...")
        for i in range(num_frames):
            # Capture frame as a NumPy array (RGB)
            frame = cam.capture_array()
            frames.append(frame)
            print(f"Captured frame {i+1}/{num_frames}")
            
            if interval > 0 and i < num_frames - 1:
                time.sleep(interval)
                
    finally:
        cam.stop()
        cam.close()
        print("Camera closed.")
        
    # Convert the list of frames into a single multi-dimensional NumPy array
    # Shape will be: (num_frames, height, width, 3)
    frame_stack = np.array(frames)
    
    # Save the entire stack into one multi-page TIFF file
    print(f"Saving multi-page stack to {filename}...")
    tifffile.imwrite(filename, frame_stack, photometric='rgb')
    print("Saved successfully!")
    
def capture_video(num_frames=100, filename="output_video.mp4", fps=10):
    """
    Captures frames using Picamera2 and writes them directly to an MP4 video 
    file to prevent out-of-memory (exit code -9) crashes.
    
    Parameters:
    - num_frames (int): Total number of frames to capture.
    - filename (str): Path and name of the output MP4 file.
    - fps (int/float): Frames per second for the output video.
    """
    cam = Picamera2()
    
    # Configure for still image capture (full resolution)
    config = cam.create_still_configuration()
    cam.configure(config)
    
    print("Starting camera and warming up...")
    cam.start()
    time.sleep(2)  # Allow auto-exposure and white balance to settle
    
    out = None
    
    try:
        print(f"Capturing and writing {num_frames} frames...")
        for i in range(num_frames):
            # Capture frame as a NumPy array (RGB)
            frame = cam.capture_array()
            
            # Initialize VideoWriter on the very first frame once dimensions are known
            if out is None:
                height, width, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                print(f"Video writer initialized ({width}x{height})")
            
            # Convert RGB to BGR for OpenCV and write directly to disk
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            print(f"Processed frame {i+1}/{num_frames}")
            
    finally:
        if out is not None:
            out.release()
        cam.stop()
        cam.close()
        print("Camera closed and video saved successfully!")

# Example usage:
capture_video(num_frames=100, filename="output.mp4", fps=15)
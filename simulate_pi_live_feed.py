import httpx
import asyncio
import time
import cv2
import numpy as np
import os
from datetime import datetime

# --- IMPORTANT: Configure your Render service URL here ---
BASE_URL = "https://zero07-yolo-weapon-api.onrender.com"

# --- URL for sending frames ---
STREAM_FRAME_URL = f"{BASE_URL}/stream_frame/"

# --- Path to your simulation video file ---
# Place your video file (e.g., test_footage.mp4) in the same directory as this script,
# or provide the full path to it.
SIMULATION_VIDEO_PATH = "simulation_video.mp4" 

async def send_frame_to_backend(frame_bytes):
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {'file': ('frame.jpg', frame_bytes, 'image/jpeg')}
            response = await client.post(STREAM_FRAME_URL, files=files)
            response.raise_for_status()
            # print(f"Frame sent successfully! Response: {response.json()}") # Uncomment for detailed log
    except httpx.RequestError as e:
        print(f"An error occurred while sending frame to {STREAM_FRAME_URL}: {e}")
    except httpx.HTTPStatusError as e:
        print(f"Server returned an error status {e.response.status_code}: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred while sending frame: {e}")

async def main_live_feed_simulator():
    print("Starting Live Feed Simulator...")
    print(f"Frames will be sent to: {STREAM_FRAME_URL}")

    # Check if the video file exists
    if not os.path.exists(SIMULATION_VIDEO_PATH):
        print(f"Error: Simulation video file '{SIMULATION_VIDEO_PATH}' not found.")
        print("Please place a video file (e.g., test_footage.mp4) in the same directory as this script.")
        print("Exiting simulator.")
        return

    cap = cv2.VideoCapture(SIMULATION_VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{SIMULATION_VIDEO_PATH}'.")
        print("Please ensure the video file is valid and OpenCV can read it.")
        return

    # Get video properties to control frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Avoid division by zero if fps is not available
        fps = 30 # Default to 30 FPS if not specified
    delay_per_frame = 1.0 / fps # Calculate delay to maintain original video FPS

    print(f"Reading video: {SIMULATION_VIDEO_PATH} at {fps:.2f} FPS")

    while True:
        ret, frame = cap.read()

        # If the video ends, loop back to the beginning
        if not ret:
            print("End of video. Looping back to start.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Set video position to the beginning
            ret, frame = cap.read() # Read the first frame again
            if not ret: # If still can't read, something is wrong
                print("Error: Could not loop video. Exiting.")
                break

        # Resize frame to a consistent size (e.g., 640x480) for performance and consistency
        # This is optional, but often good practice for streaming to an API
        frame = cv2.resize(frame, (640, 480)) 

        # Encode the frame to JPEG bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Could not encode frame to JPEG.")
            await asyncio.sleep(0.1) # Short delay to prevent tight loop on error
            continue
        
        frame_bytes = buffer.tobytes()

        await send_frame_to_backend(frame_bytes)
        
        # Maintain the original video's frame rate
        await asyncio.sleep(delay_per_frame) 

if __name__ == "__main__":
    # Ensure you have opencv-python and httpx installed:
    # pip install opencv-python httpx
    asyncio.run(main_live_feed_simulator())
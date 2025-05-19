import requests
import cv2
import numpy as np
import os
import time
import io
import json # Import json for pretty printing

# Define the URL of your running FastAPI endpoint
# Make sure this matches where your uvicorn server is running
API_URL = "http://127.0.0.1:8000/detect_object/"

def send_image_to_api(image_path):
    """
    Reads an image file and sends it to the FastAPI API for detection.
    Prints the detection results.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        # Open the image file in binary read mode
        with open(image_path, "rb") as f:
            # Prepare the file for the POST request
            # The key 'file' must match the parameter name in the FastAPI endpoint (file: UploadFile = File(...))
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")} # Adjust content type if needed

            print(f"Sending image: {os.path.basename(image_path)} to API...")
            response = requests.post(API_URL, files=files)

            # Check the response status code
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Parse and return the JSON response
            return response.json()

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {API_URL}")
        print("Please ensure your FastAPI server is running and accessible.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error sending image to API: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while sending image {image_path}: {e}")
        return None


def send_frame_to_api(frame, frame_name="frame.jpg"):
    """
    Encodes a video frame (NumPy array) and sends it to the FastAPI API for detection.
    Returns the detection results.
    """
    try:
        # Encode the frame as a JPEG image in memory
        # You can use other formats like .png if needed, but jpeg is usually smaller and faster
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            print(f"Error: Could not encode frame to JPEG for {frame_name}")
            return None

        # Convert the buffer to bytes
        byte_frame = io.BytesIO(buffer).read()

        # Prepare the file for the POST request
        files = {"file": (frame_name, byte_frame, "image/jpeg")} # Adjust content type if needed

        # print(f"Sending frame: {frame_name} to API...") # Uncomment for detailed frame logs
        response = requests.post(API_URL, files=files)

        # Check the response status code
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Parse and return the JSON response
        return response.json()

    except requests.exceptions.ConnectionError:
         print(f"Error: Could not connect to API at {API_URL}")
         print("Please ensure your FastAPI server is running and accessible.")
         return None
    except requests.exceptions.RequestException as e:
        print(f"Error sending frame to API ({frame_name}): {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing and sending frame {frame_name}: {e}")
        return None


def process_video(video_path, frames_to_skip=0):
    """
    Reads a video file frame by frame and sends each frame to the API for detection.
    Displays timestamp and detection info when a weapon is detected.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Optional: get total frame count

    if frame_rate <= 0:
         print("Warning: Could not get valid frame rate from video. Timestamps may be inaccurate. Using 30 fps.")
         frame_rate = 30.0 # Default to 30 fps if frame rate is zero or negative


    frame_count = 0
    print(f"Processing video: {os.path.basename(video_path)}")
    print(f"Frame Rate: {frame_rate:.2f} fps")
    print("-" * 30)


    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break # End of video or error reading frame

        frame_count += 1

        # Skip frames if specified
        # The logic (frame_count - 1) % (frames_to_skip + 1) == 0 processes frame 1, frame 1+skip+1, etc.
        if frames_to_skip > 0 and (frame_count - 1) % (frames_to_skip + 1) != 0:
             # print(f"Skipping frame {frame_count}") # Uncomment for detailed frame logs
             continue

        # Calculate timestamp in seconds
        # Timestamps are 0-based, so use frame_count - 1
        timestamp_seconds = (frame_count - 1) / frame_rate

        # Format timestamp as HH:MM:SS.ms
        hours = int(timestamp_seconds // 3600)
        minutes = int((timestamp_seconds % 3600) // 60)
        seconds = int(timestamp_seconds % 60)
        # Get milliseconds with 3 digits
        milliseconds = int((timestamp_seconds - int(timestamp_seconds)) * 1000)
        timestamp_str = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


        print(f"Processing frame {frame_count} (Timestamp: {timestamp_str})...") # Progress indicator

        # Send frame to API for detection
        frame_name = f"video_{os.path.basename(video_path)}_frame_{frame_count}_{timestamp_str}.jpg" # Add timestamp to name
        detection_results = send_frame_to_api(frame, frame_name) # Send frame to API


        if detection_results:
            # Process the detection results for the frame
            detections = detection_results.get("detections", [])
            # Check if any detected object is a weapon (firearm, axe, or knife)
            weapon_detections_in_frame = [d for d in detections if d['class_name'] in ['firearm', 'axe', 'knife']]

            if weapon_detections_in_frame:
                 # Print timestamp and weapon detection info if any weapon was detected
                 print(f"\n{'='*40}")
                 print(f"!!! WEAPON DETECTED !!!")
                 print(f"Timestamp: {timestamp_str}")
                 print(f"Frame: {frame_count}")
                 print(f"Video: {os.path.basename(video_path)}")
                 print("Detected weapons in this frame:")
                 for weapon_det in weapon_detections_in_frame:
                     print(f"  - {weapon_det['class_name']} (Confidence: {weapon_det['confidence']:.2f})")
                 print(f"{'='*40}\n")
                 # Note: The saving of the original frame and handler crops happens on the SERVER side (in main.py)
                 # if the confidence is above the threshold set in main.py (currently 0.6)


        # Optional: Add a small delay if needed to control processing speed (simulates real-time or slower)
        # time.sleep(0.01) # Sleep for 10 milliseconds per frame


    cap.release()
    print(f"\nFinished processing video: {os.path.basename(video_path)}")
    print("-" * 30)


if __name__ == "__main__":
    # --- How to Use ---
    # Choose ONE of the sections below (IMAGE or VIDEO) to uncomment and run at a time.
    # Update the file paths to point to actual files on your computer.

    # --- Option 1: Test with a single IMAGE file ---
    # Uncomment the lines below to test with an image
    # image_file_path = r"C:\path\to\your\test_image.jpg" # <--- UPDATE THIS PATH
    #
    # print("--- Testing with Image ---")
    # results = send_image_to_api(image_file_path)
    # if results:
    #      print("\n--- Full Detection Results for Image ---")
    #      # Use json.dumps for pretty printing the JSON response
    #      print(json.dumps(results, indent=4))
    # print("-" * 30)


    # --- Option 2: Test with a VIDEO file ---
    # Uncomment the lines below to test with a video
    video_file_path = r"C:\Users\olakl\Downloads\X2Twitter.com_tLRKNTeCWlLWl8Au_720p.mp4" # <--- UPDATE THIS PATH

    # Adjust frames_to_skip: 0 = process every frame, 1 = process every other frame, etc.
    # Set higher to process videos faster for initial testing.
    frames_to_skip = 0
    
    print("--- Testing with Video ---")
    process_video(video_file_path, frames_to_skip=frames_to_skip)


    print("\nClient script finished.")
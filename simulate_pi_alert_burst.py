import httpx
import asyncio
import json
import time
from datetime import datetime
import random
import uuid # For generating unique IDs for dummy image URLs
import os # For checking if the script exists

# --- IMPORTANT: Configure your Render service URL here ---
BASE_URL = "https://zero07-yolo-weapon-api.onrender.com/detection_alert/" 

# --- Function to simulate burst image capture and upload ---
# In a real Raspberry Pi, this would:
# 1. Use the camera library (e.g., picamera2, OpenCV) to capture 4-5 frames rapidly.
# 2. Save each frame temporarily.
# 3. Upload each frame to a cloud storage service (e.g., AWS S3, Google Cloud Storage, Cloudinary).
# 4. Return the public URLs of the uploaded images.
def simulate_burst_capture_and_upload(num_images=5):
    image_urls = []
    print(f"Simulating capture of {num_images} images...")
    for i in range(num_images):
        # --- REPLACE THIS WITH ACTUAL CAMERA CAPTURE LOGIC ---
        # e.g., capture_frame_to_buffer()
        
        # --- REPLACE THIS WITH ACTUAL CLOUD UPLOAD LOGIC ---
        # e.g., upload_to_s3(frame_data, f"alert_image_{uuid.uuid4()}.jpg")
        # For simulation, we generate dummy URLs from Picsum.photos
        # Using a random ID to get different images each time
        dummy_url = f"https://picsum.photos/400/300?random={uuid.uuid4()}" 
        image_urls.append(dummy_url)
        # Simulate a small delay between captures (even if not microsecond-precise)
        time.sleep(0.05) # 50 milliseconds delay
    print("Simulation: Images captured and dummy URLs generated.")
    return image_urls

async def send_alert(alert_data):
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(BASE_URL, json=alert_data)
            response.raise_for_status()
            print(f"Alert sent successfully! Response: {response.json()}")
    except httpx.RequestError as e:
        print(f"An error occurred while requesting {BASE_URL}: {e}")
        print("Failed to send alert. Check if the server is running and the URL is correct.")
    except httpx.HTTPStatusError as e:
        print(f"Server returned an error status {e.response.status_code}: {e.response.text}")
        print("Failed to send alert. Render returned an error.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

async def main():
    while True:
        # Simulate detection of a weapon
        detected_type = "weapon" 

        # Simulate some dummy coordinates and confidence
        coords = [round(random.uniform(0, 1), 2) for _ in range(4)] # [x1, y1, x2, y2]
        confidence = round(random.uniform(0.7, 0.99), 2)

        # --- Trigger burst capture when weapon is detected ---
        burst_images = simulate_burst_capture_and_upload(num_images=5)

        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "location": "Main Entrance Camera 1",
            "detection_type": detected_type,
            "confidence": confidence,
            "coordinates": coords,
            "alert_images": burst_images # Include the list of image URLs
        }

        print(f"\nSimulating alert: {json.dumps(alert_data, indent=2)}")
        await send_alert(alert_data)

        # Wait for a few seconds before sending the next alert
        await asyncio.sleep(random.randint(5, 15)) # Send alerts every 5-15 seconds

if __name__ == "__main__":
    print("Starting Raspberry Pi Alert Simulator with Burst Images...")
    print(f"Alerts will be sent to: {BASE_URL}")
    asyncio.run(main())
import httpx
import asyncio
import json
import time
from datetime import datetime
import random

# --- UPDATE THIS BASE_URL with your Render service URL ---
# Make sure to append '/detection_alert/' to the base URL
BASE_URL = "https://zero07-yolo-weapon-api.onrender.com/detection_alert/" 

async def send_alert(alert_data):
    try:
        # Create an AsyncClient to make HTTP requests
        # Using a timeout to ensure the request doesn't hang indefinitely
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(BASE_URL, json=alert_data)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            print(f"Alert sent successfully! Response: {response.json()}")
    except httpx.RequestError as e:
        # Handles network-related errors (e.g., DNS resolution failure, connection refused)
        print(f"An error occurred while requesting {BASE_URL}: {e}")
        print("Failed to send alert. Check if the server is running and the URL is correct.")
    except httpx.HTTPStatusError as e:
        # Handles HTTP status errors (e.g., 404 Not Found, 500 Internal Server Error)
        print(f"Server returned an error status {e.response.status_code}: {e.response.text}")
        print("Failed to send alert. Render returned an error.")
    except Exception as e:
        # Catches any other unexpected errors
        print(f"An unexpected error occurred: {e}")

async def main():
    while True:
        # Simulate different detection types
        detection_types = ["weapon", "person_of_interest", "unusual_activity", "package_left"]
        detected_type = random.choice(detection_types)

        # Simulate some dummy coordinates and confidence
        coords = [round(random.uniform(0, 1), 2) for _ in range(4)] # [x1, y1, x2, y2]
        confidence = round(random.uniform(0.6, 0.99), 2)

        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "location": "Main Entrance Camera 1",
            "detection_type": detected_type,
            "confidence": confidence,
            "coordinates": coords,
            "image_url": "https://example.com/some_cctv_footage.jpg" # Dummy URL
        }

        print(f"\nSimulating alert: {json.dumps(alert_data, indent=2)}")
        await send_alert(alert_data)

        # Wait for a few seconds before sending the next alert
        await asyncio.sleep(random.randint(5, 15)) # Send alerts every 5-15 seconds

if __name__ == "__main__":
    print("Starting Raspberry Pi Alert Simulator...")
    print(f"Alerts will be sent to: {BASE_URL}")
    asyncio.run(main())
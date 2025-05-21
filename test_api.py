import requests
import os

# --- Configuration ---
# Replace with the actual URL of your deployed Render service + the endpoint path
# Use the URL you just found: https://zero07-yolo-weapon-api.onrender.com
API_URL = "https://zero07-yolo-weapon-api.onrender.com/detect_object/"

# Replace with the actual path to a test image file on your computer
# Make sure this image file exists!
IMAGE_PATH = "C:/Users/olakl/Desktop/man with gun.jpeg" # e.g., "C:/Users/YourName/Pictures/test.jpg"

# --- Check if the image file exists ---
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image file not found at {IMAGE_PATH}")
    print("Please update the IMAGE_PATH variable with the correct path to your test image.")
    exit()

# --- Send the POST request ---
print(f"Sending image '{IMAGE_PATH}' to API endpoint: {API_URL}")

try:
    # Open the image file in binary read mode
    with open(IMAGE_PATH, 'rb') as f:
        # Create the files dictionary for the POST request
        # The key 'file' must match the parameter name in your FastAPI endpoint (file: UploadFile = File(...))
        files = {'file': (os.path.basename(IMAGE_PATH), f)}

        # Send the POST request
        response = requests.post(API_URL, files=files)

    # --- Process the response ---
    print(f"API Response Status Code: {response.status_code}")

    if response.status_code == 200:
        # Request was successful, print the JSON response
        detections = response.json()
        print("Detection Results (JSON):")
        # You can print the whole JSON or iterate through detections
        print(json.dumps(detections, indent=2)) # Pretty print the JSON

    else:
        # Request failed, print the error message from the response body
        print(f"API Request Failed: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the API request: {e}")
    print("Please check if the Render service is running and the URL is correct.")

print("\nScript finished.")
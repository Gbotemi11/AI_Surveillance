import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import io
import numpy as np
from PIL import Image
import os
import cv2
import time # To generate unique filenames
import shutil # To remove and recreate directories if needed
import json # Import json for potential detailed error logging

# Define the path to your trained model - NOW READ FROM ENVIRONMENT VARIABLE
# Ensure you set a MODEL_PATH_ENV environment variable in your deployment environment
# For local testing, you can set this variable in your terminal before running (e.g., set MODEL_PATH_ENV=C:\Users\olakl\runs\detect\train10\weights\best.pt)
MODEL_PATH = os.environ.get('MODEL_PATH_ENV')

# Check if the MODEL_PATH environment variable is set
if not MODEL_PATH:
    print("Server CRITICAL ERROR: MODEL_PATH_ENV environment variable not set.")
    # In a production environment, you might want to exit here
    # raise RuntimeError("MODEL_PATH_ENV environment variable not set.")
    # For now, we'll allow it to continue but model loading will fail

# Define directories for saving detected images - NOTE: These will be ephemeral on standard Render
DETECTED_WEAPONS_DIR = "detected_weapons"
WEAPON_HANDLERS_DIR = "weapon_handlers"

# Create the directories if they don't exist (Note: Will be created on Render server each deploy)
os.makedirs(DETECTED_WEAPONS_DIR, exist_ok=True)
os.makedirs(WEAPON_HANDLERS_DIR, exist_ok=True)

# Create a FastAPI application instance
app = FastAPI()

# Variable to hold the loaded model
model = None

@app.on_event("startup")
async def load_model():
    """Load the model when the FastAPI application starts."""
    global model
    if not MODEL_PATH:
        print("Server: Skipping model load because MODEL_PATH_ENV is not set.")
        return # Don't attempt to load if path is missing

    try:
        print(f"Server: Attempting to load model from {MODEL_PATH}...")
        # Set verbose=False to reduce model loading output if desired
        model = YOLO(MODEL_PATH)
        print(f"Server: Model loaded successfully from {MODEL_PATH}. Ready for predictions!")
    except Exception as e:
        print(f"Server CRITICAL ERROR: Could not load model from {MODEL_PATH}. Details: {e}")
        # In a real application, this should probably prevent the server from starting successfully
        # raise RuntimeError(f"Failed to load model: {e}")


@app.post("/detect_object/")
async def detect_object(file: UploadFile = File(...)):
    """
    Receives an image file (or video frame), performs object detection,
    and saves the image and handler crops if a weapon is detected above a threshold.
    Returns ALL detection results.
    """
    # Check if the model was loaded successfully at startup
    if model is None:
         print("Server Error: Model is not loaded. Check server startup logs or MODEL_PATH_ENV.")
         # Return a 503 Service Unavailable if the model isn't ready
         raise HTTPException(status_code=503, detail="Model not loaded. Check server startup logs or MODEL_PATH_ENV environment variable.")

    try:
        # Read the image file
        image_data = await file.read()

        # Convert image data to a format OpenCV can use (NumPy array)
        image_np = np.frombuffer(image_data, np.uint8)
        image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR) # Use imdecode for flexibility

        if image_cv2 is None:
             print(f"Server Error: Could not decode image file: {file.filename}")
             raise HTTPException(status_code=400, detail="Could not decode image file. Ensure it's a valid image format.")

        # Perform inference using the loaded model
        results = model.predict(source=image_cv2, save=False, save_txt=False, conf=0.25) # You can adjust this prediction confidence

        # Process results and check for weapons/persons for potential saving
        detections = []
        weapon_detected_for_saving = False # Flag for saving based on confidence threshold
        person_detections_for_saving = [] # Store person detections meeting confidence for saving

        # Assuming predict returns a list of results, one for our single image input
        if results:
            r = results[0] # Get the results for the first (and only) image

            # Extract bounding boxes, confidences, and class IDs
            # Use .cpu().tolist() to ensure tensors are on CPU and converted to lists
            boxes_xyxy = r.boxes.xyxy.cpu().tolist() # xyxy format [x1, y1, x2, y2]
            confidences = r.boxes.conf.cpu().tolist()
            class_ids = r.boxes.cls.cpu().tolist()
            # Ensure class_ids are integers for lookup, use .get() for safety if id is unexpected
            class_names = [model.names.get(int(cid), "unknown_class") for cid in class_ids]

            # Iterate through detections in this result object
            for i in range(len(boxes_xyxy)):
                class_name = class_names[i]
                box = boxes_xyxy[i]
                confidence = confidences[i]

                # Store ALL detection details for the JSON response (regardless of saving threshold)
                detections.append({
                    "box": box,
                    "confidence": confidence,
                    "class_id": int(class_ids[i]),
                    "class_name": class_name
                })

                # --- Check for saving based on confidence threshold ---
                SAVING_WEAPON_CONF_THRESHOLD = 0.45 # Lowered threshold based on video results
                if class_name in ['firearm', 'axe', 'knife'] and confidence >= SAVING_WEAPON_CONF_THRESHOLD:
                    weapon_detected_for_saving = True
                    print(f"Server: Weapon detected above saving threshold: {class_name} with confidence {confidence:.2f}")

                SAVING_PERSON_CONF_THRESHOLD = 0.45 # Match person threshold or adjust separately
                if class_name == 'person' and confidence >= SAVING_PERSON_CONF_THRESHOLD:
                    person_detections_for_saving.append({
                         "box": box,
                         "confidence": confidence,
                         "class_name": class_name
                    })
                    print(f"Server: Person detected above saving threshold with confidence {confidence:.2f}")


        # --- Saving Logic ---
        # NOTE: Files saved here will be LOST on standard Render Web Services after restarts/redeployments
        if weapon_detected_for_saving:
            timestamp = int(time.time())
            original_filename_clean = "".join([c if c.isalnum() or c in (' ._-') else '_' for c in file.filename]) if file.filename else f"uploaded_image"
            base_filename, file_extension = os.path.splitext(original_filename_clean)
            if not file_extension or file_extension.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                 file_extension = '.jpg'

            unique_id = os.urandom(4).hex()
            unique_filename_base = f"{base_filename}_{timestamp}_{unique_id}"

            try:
                img_with_boxes = image_cv2.copy()
                for det in detections: # Use all detections for drawing
                    box = det["box"]
                    class_name = det["class_name"]
                    confidence = det["confidence"]
                    x1, y1, x2, y2 = map(int, box)

                    color = (0, 255, 0)
                    thickness = 2
                    font_scale = 0.5
                    font_thickness = 1

                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

                    label = f"{class_name}: {confidence:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    text_pos = (x1, y1 - 10 if y1 - 10 > text_height else y1 + 10)
                    text_pos = (max(0, text_pos[0]), max(text_height, text_pos[1]))

                    cv2.putText(img_with_boxes, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)

                weapon_image_path = os.path.join(DETECTED_WEAPONS_DIR, f"{unique_filename_base}_detections{file_extension}")
                cv2.imwrite(weapon_image_path, img_with_boxes)
                print(f"Server: Saved weapon detected image with boxes: {weapon_image_path}")

            except Exception as save_e:
                 print(f"Server Error: Could not save weapon image {weapon_image_path} with boxes. Details: {save_e}")


            if person_detections_for_saving:
                for i, person_det in enumerate(person_detections_for_saving):
                    p_box = person_det["box"]
                    padding = 20
                    h, w = image_cv2.shape[:2]

                    x_min, y_min, x_max, y_max = map(int, p_box)
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w - 1, x_max + padding)
                    y_max = min(h - 1, y_max + padding)

                    if y_max > y_min and x_max > x_min:
                        person_crop = image_cv2[y_min:y_max, x_min:x_max]

                        handler_image_path = os.path.join(WEAPON_HANDLERS_DIR, f"{unique_filename_base}_handler_{i}{file_extension}")

                        try:
                            cv2.imwrite(handler_image_path, person_crop)
                            print(f"Server: Saved handler crop: {handler_image_path}")
                        except Exception as save_e:
                            print(f"Server Error: Could not save handler crop {handler_image_path}. Details: {save_e}")
                    else:
                         print(f"Server Warning: Skipping save for person detection {i}. Invalid cropping coordinates ({x_min},{y_min},{x_max},{y_max}).")

            elif weapon_detected_for_saving:
                 print("Server: Weapon detection saved, but no person detection met the saving threshold in this frame to save handler crop.")


        # Return the JSON response with ALL detection details
        return JSONResponse(content={"detections": detections})

    except Exception as e:
        print(f"Server Error: An unexpected error occurred during object detection or saving: {e}")
        return JSONResponse(content={"error": f"An internal server error occurred: {e}", "detections": detections if 'detections' in locals() else []}, status_code=500)


# Optional: Add a root endpoint for basic testing
@app.get("/")
async def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "YOLOv8 Object Detection API is running. Send POST requests with image files to /detect_object/ to detect objects."}

# The if __name__ == "__main__": block is primarily for local development.
# Production servers like Gunicorn will call the `app` instance directly.
if __name__ == "__main__":
    print("--- Running locally with uvicorn ---")
    # When running locally, you might still want to use a local path
    # Or you can set the MODEL_PATH_ENV environment variable in your terminal
    # e.g., set MODEL_PATH_ENV=C:\Users\olakl\runs\detect\train10\weights\best.pt
    # uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) # Use 127.0.0.1 for local only
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) # Use 0.0.0.0 for local network access
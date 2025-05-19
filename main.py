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

# Define the path to your trained model
# Make sure this path is correct for your system
MODEL_PATH = r"C:\Users\olakl\runs\detect\train10\weights\best.pt" # Ensure this points to your latest best model (from train10)

# Define directories for saving detected images
# These directories will be created in the same folder as your main.py script
DETECTED_WEAPONS_DIR = "detected_weapons"
WEAPON_HANDLERS_DIR = "weapon_handlers"

# --- Optional: Clean previous test results if needed ---
# Uncomment the lines below if you want to clear the saving folders each time you run the app
# Warning: This will delete all previously saved images in these folders on startup!
# if os.path.exists(DETECTED_WEAPONS_DIR):
#     print(f"Server: Clearing directory: {DETECTED_WEAPONS_DIR}")
#     shutil.rmtree(DETECTED_WEAPONS_DIR)
# if os.path.exists(WEAPON_HANDLERS_DIR):
#     print(f"Server: Clearing directory: {WEAPON_HANDLERS_DIR}")
#     shutil.rmtree(WEAPON_HANDLERS_DIR)

# Create the directories if they don't exist
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
    try:
        print(f"Server: Attempting to load model from {MODEL_PATH}...")
        # Set verbose=False to reduce model loading output if desired
        model = YOLO(MODEL_PATH)
        print(f"Server: Model loaded successfully from {MODEL_PATH}. Ready for predictions!")
    except Exception as e:
        print(f"Server CRITICAL ERROR: Could not load model from {MODEL_PATH}. Details: {e}")
        # In a real application, you might want to handle this more gracefully, e.g., log and exit.
        raise RuntimeError(f"Failed to load model: {e}")


@app.post("/detect_object/")
async def detect_object(file: UploadFile = File(...)):
    """
    Receives an image file (or video frame), performs object detection,
    and saves the image and handler crops if a weapon is detected above a threshold.
    Returns ALL detection results.
    """
    # Check if the model was loaded successfully at startup
    if model is None:
         print("Server Error: Model is not loaded. Check server startup logs.")
         raise HTTPException(status_code=503, detail="Model not loaded. Check server startup logs.")

    # --- Basic file size limit (optional, uncomment to enable) ---
    # MAX_FILE_SIZE_MB = 20
    # try:
    #     file.file.seek(0, os.SEEK_END)
    #     file_size = file.file.tell()
    #     file.file.seek(0) # Seek back to the beginning of the file
    #     if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
    #          print(f"Server Warning: Uploaded file {file.filename} exceeds size limit of {MAX_FILE_SIZE_MB}MB")
    #          raise HTTPException(status_code=413, detail=f"File size exceeds limit of {MAX_FILE_SIZE_MB}MB")
    # except Exception as e:
    #      print(f"Server Error: Could not check file size for {file.filename}. Details: {e}")
    #      # Continue processing, but log the error, or raise exception if size check is critical
    #      # raise HTTPException(status_code=500, detail=f"Could not process file size: {e}")


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
        # Use the loaded model object 'model' instead of calling YOLO() again
        # Adding conf=0.25 here means detections below 0.25 are filtered out by the model itself before processing
        results = model.predict(source=image_cv2, save=False, save_txt=False, conf=0.25) # You can adjust this prediction confidence

        # Process results and check for weapons/persons for potential saving
        detections = []
        weapon_detected_for_saving = False # Flag for saving based on confidence threshold
        person_detections_for_saving = [] # Store person detections meeting confidence for saving

        # Assuming predict returns a list of results, one for our single image input
        if results:
            r = results[0] # Get the results for the first (and only) image

            # Extract bounding boxes, confidences, and class IDs
            boxes_xyxy = r.boxes.xyxy.tolist() # xyxy format [x1, y1, x2, y2]
            confidences = r.boxes.conf.tolist()
            class_ids = r.boxes.cls.tolist()
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
                # WEAPON Check (Adjust this threshold to control when to save)
                SAVING_WEAPON_CONF_THRESHOLD = 0.45 # Lowered threshold based on video results
                if class_name in ['firearm', 'axe', 'knife'] and confidence >= SAVING_WEAPON_CONF_THRESHOLD: # Using >= to include threshold value
                    weapon_detected_for_saving = True
                    # Print message on server console when weapon meets saving threshold
                    print(f"Server: Weapon detected above saving threshold: {class_name} with confidence {confidence:.2f}")

                # PERSON Check (Store persons if they also meet a threshold, for handler saving)
                SAVING_PERSON_CONF_THRESHOLD = 0.45 # Match person threshold or adjust separately
                if class_name == 'person' and confidence >= SAVING_PERSON_CONF_THRESHOLD: # Using >= to include threshold value
                    person_detections_for_saving.append({
                         "box": box,
                         "confidence": confidence,
                         "class_name": class_name # Store name for clarity
                    })
                    # Print message on server console when person meets saving threshold
                    print(f"Server: Person detected above saving threshold with confidence {confidence:.2f}")


        # --- Saving Logic ---
        # This block runs only if a weapon was detected ABOVE or AT the saving threshold
        if weapon_detected_for_saving:
            # Generate a unique filename (timestamp + original filename + unique ID)
            timestamp = int(time.time())
            # Clean the original filename to be safe for file paths
            original_filename_clean = "".join([c if c.isalnum() or c in (' ._-') else '_' for c in file.filename]) if file.filename else f"uploaded_image"
            # Ensure a default extension if missing or invalid
            base_filename, file_extension = os.path.splitext(original_filename_clean)
            if not file_extension or file_extension.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                 file_extension = '.jpg' # Default to jpg

            # Add a short unique ID in case multiple frames processed quickly have the same timestamp base filename
            unique_id = os.urandom(4).hex() # 8 random hex characters
            # Add frame count if available from client (optional advanced)
            # frame_info = file.filename.split('_frame_')[-1].split('.')[0] if '_frame_' in file.filename else ""
            # unique_filename_base = f"{base_filename}_{timestamp}_{frame_info}_{unique_id}" if frame_info else f"{base_filename}_{timestamp}_{unique_id}"
            unique_filename_base = f"{base_filename}_{timestamp}_{unique_id}"


            # --- Save the original image with bounding boxes drawn ---
            try:
                # Draw bounding boxes on the image before saving
                img_with_boxes = image_cv2.copy()
                # Assuming 'detections' contains all detections returned by the API
                for det in detections:
                    box = det["box"]
                    class_name = det["class_name"]
                    confidence = det["confidence"]
                    x1, y1, x2, y2 = map(int, box)

                    # Define color and thickness (you can customize this)
                    color = (0, 255, 0) # Green color for bounding box
                    thickness = 2
                    font_scale = 0.5
                    font_thickness = 1

                    # Draw rectangle
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)

                    # Put label and confidence near the box
                    label = f"{class_name}: {confidence:.2f}"
                    # Calculate text size to position it correctly
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    # Position the text slightly above the box or inside if box is small
                    text_pos = (x1, y1 - 10 if y1 - 10 > text_height else y1 + 10)
                    # Ensure text is not out of image bounds
                    text_pos = (max(0, text_pos[0]), max(text_height, text_pos[1]))


                    cv2.putText(img_with_boxes, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)


                # Save the image with drawn boxes
                weapon_image_path = os.path.join(DETECTED_WEAPONS_DIR, f"{unique_filename_base}_detections{file_extension}") # Added _detections to filename
                cv2.imwrite(weapon_image_path, img_with_boxes)
                print(f"Server: Saved weapon detected image with boxes: {weapon_image_path}")

            except Exception as save_e:
                 print(f"Server Error: Could not save weapon image {weapon_image_path} with boxes. Details: {save_e}")


            # --- Save crops of detected persons (handlers) ---
            # This runs if a weapon was saved and at least one person also met their saving threshold
            if person_detections_for_saving:
                for i, person_det in enumerate(person_detections_for_saving):
                    p_box = person_det["box"]
                    # Ensure bounding box coordinates are integers for cropping
                    # Add padding around the bounding box (optional, can adjust)
                    padding = 20 # pixels
                    h, w = image_cv2.shape[:2] # Get image dimensions

                    # Get coordinates and apply padding, ensuring they are within image bounds
                    x_min, y_min, x_max, y_max = map(int, p_box)
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w - 1, x_max + padding) # Ensure max coords are within bounds
                    y_max = min(h - 1, y_max + padding)


                    # Crop the person region from the original image
                    # Ensure coordinates define a valid region (height and width > 0)
                    if y_max > y_min and x_max > x_min:
                        person_crop = image_cv2[y_min:y_max, x_min:x_max]

                        # Generate a unique filename for the cropped person
                        handler_image_path = os.path.join(WEAPON_HANDLERS_DIR, f"{unique_filename_base}_handler_{i}{file_extension}")

                        # Save the cropped person image
                        try:
                            cv2.imwrite(handler_image_path, person_crop)
                            print(f"Server: Saved handler crop: {handler_image_path}")
                        except Exception as save_e:
                            print(f"Server Error: Could not save handler crop {handler_image_path}. Details: {save_e}")
                    else:
                         print(f"Server Warning: Skipping save for person detection {i}. Invalid cropping coordinates ({x_min},{y_min},{x_max},{y_max}).")

            elif weapon_detected_for_saving: # Weapon saved, but no person met the saving threshold in this frame
                 print("Server: Weapon detection saved, but no person detection met the saving threshold in this frame to save handler crop.")


        # Return the JSON response with ALL detection details (even those below saving threshold)
        # This JSON is what the web front-end will receive to draw bounding boxes
        return JSONResponse(content={"detections": detections})

    except Exception as e:
        print(f"Server Error: An unexpected error occurred during object detection or saving: {e}")
        # Return an error response with details, including any detections found before the error
        return JSONResponse(content={"error": f"An internal server error occurred: {e}", "detections": detections if 'detections' in locals() else []}, status_code=500)


# Optional: Add a root endpoint for basic testing
@app.get("/")
async def read_root():
    """Root endpoint to check if the API is running."""
    return {"message": "YOLOv8 Object Detection API is running. Send POST requests with image files to /detect_object/ to detect objects."}

if __name__ == "__main__":
    # Command to run the FastAPI application using uvicorn
    # host="127.0.0.1" means it's only accessible from your local machine.
    # To make it accessible from other machines on your local network, you might need to change host="0.0.0.0" (be cautious).
    # reload=True is useful for development, automatically restarts server on code changes.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
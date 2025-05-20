from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import cv2
import numpy as np
from typing import List, Dict
import json # Make sure json is imported for broadcast_message

# Import YOLO from ultralytics
# Ensure you have ultralytics installed: pip install ultralytics
from ultralytics import YOLO
import torch # <-- ADDED FOR MODEL LOADING FIX

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="API for detecting objects using a YOLOv8 model.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Allows requests from any origin. Adjust origins in a production environment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Global variable for the model ---
model = None
model_path = None

# --- Startup Event to Load Model ---
@app.on_event("startup")
async def load_model():
    global model, model_path
    # Get model path from environment variable, default to local if not set
    # On Render, MODEL_PATH_ENV will be 'weights/best.pt'
    # Locally, you might set it to 'best.pt' or 'weights/best.pt' if it's in a 'weights' folder
    model_path_env = os.getenv('MODEL_PATH_ENV')

    if model_path_env:
        model_path = model_path_env
        print(f"Server: Attempting to load model from environment variable path: {model_path}")
    else:
        # Fallback for local testing if env var not set, or if model is in root
        model_path = "best.pt" # Adjust if your model is in a 'weights/' subfolder locally
        if not os.path.exists(model_path):
            model_path = "weights/best.pt" # Check common local path
        print(f"Server: MODEL_PATH_ENV not set. Attempting to load model from default local path: {model_path}")

    try:
        # --- ADDED LINES TO HANDLE THE SAFE GLOBAL LOADING ---
        # Temporarily allowlist the DetectionModel class for safe loading
        # This is the recommended fix by the PyTorch error message.
        # This assumes Ultralytics' DetectionModel is in ultralytics.nn.tasks
        # If your Ultralytics version is different, the module path might change.
        torch.serialization.add_safe_global("ultralytics.nn.tasks", "DetectionModel")
        # --- END ADDED LINES ---

        model = YOLO(model_path)
        print(f"Server: Model loaded successfully from {model_path}. Ready for predictions!")
    except Exception as e:
        print(f"Server: Error loading model from {model_path}: {e}")
        # Log the full traceback for more details if needed during debugging
        import traceback
        traceback.print_exc()
        model = None # Ensure model is None if loading fails

# --- Root Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    if model:
        status_message = "YOLOv8 Object Detection API is running. Send POST requests with image files to `/detect_object/` to detect objects. WebSocket alerts available at `/ws`. Send alerts from Pi to `/detection_alert/`."
    else:
        status_message = "YOLOv8 Object Detection API is running, but the model failed to load. Check server logs for details."
    return JSONResponse(content={"message": status_message})

# --- Object Detection Endpoint ---
@app.post("/detect_object/")
async def detect_object(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Convert PIL Image to numpy array (OpenCV format)
    np_image = np.array(image)
    # Convert RGB to BGR for OpenCV processing if needed (though Ultralytics handles this)
    # np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    print(f"Server: Received image '{file.filename}' for detection.")

    # Perform inference
    results = model(np_image) # YOLO model expects numpy array or PIL Image

    # Process results (Ultralytics results object)
    detections = []
    for r in results:
        # Extract bounding boxes, classes, and confidence scores
        boxes = r.boxes.xyxy.cpu().numpy()  # xyxy format
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            detection = {
                "box": boxes[i].tolist(),
                "score": float(scores[i]),
                "class_id": int(classes[i]),
                "class_name": model.names[int(classes[i])] # Get class name
            }
            detections.append(detection)
    
    print(f"Server: Detected {len(detections)} objects.")
    return JSONResponse(content={"filename": file.filename, "detections": detections})


# --- WebSocket & Real-time Alert Endpoints ---

# Store active WebSocket connections
active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"WebSocket client connected: {websocket.client}")
    try:
        while True:
            # We don't expect messages from the dashboard in this basic setup,
            # but you can add logic here if needed (e.g., dashboard sending commands)
            data = await websocket.receive_text()
            print(f"Received message from client: {data}")
    except Exception as e:
        print(f"WebSocket client disconnected or error: {e}")
    finally:
        active_connections.remove(websocket)
        print(f"WebSocket client disconnected: {websocket.client}")

# Helper function to broadcast messages to all connected WebSocket clients
async def broadcast_message(message: Dict):
    # Ensure message is JSON serializable
    message_to_send = json.dumps(message)
    for connection in active_connections:
        try:
            await connection.send_text(message_to_send) # send_text is fine for JSON string
        except Exception as e:
            print(f"Error sending message to WebSocket client {connection.client}: {e}")
            # Consider removing broken connections here, but for now, simple error print

# This function will be called by your Raspberry Pi to send detection data
@app.post("/detection_alert/")
async def receive_detection_alert(detection_data: Dict):
    print(f"Received detection alert from Pi: {detection_data}")
    # Broadcast the alert to all connected WebSocket dashboards
    await broadcast_message(detection_data)
    return {"status": "alert received and broadcasted"}


# --- Local Development Server (Optional) ---
# This block is typically for local testing using `python main.py`
# Render does not use this, as it uses Gunicorn/Uvicorn to run the app.
if __name__ == "__main__":
    import uvicorn
    # Set the environment variable for local testing if you run it this way
    # This assumes 'best.pt' is in a 'weights' folder relative to your main.py
    os.environ['MODEL_PATH_ENV'] = 'weights/best.pt'
    uvicorn.run(app, host="0.0.0.0", port=8000)
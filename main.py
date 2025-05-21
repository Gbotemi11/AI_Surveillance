from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import cv2
import numpy as np
from typing import List, Dict
import json

# Import YOLO from ultralytics
from ultralytics import YOLO
import torch # Imported for model loading

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="API for detecting objects using a YOLOv8 model.",
    version="1.0.0"
)

# --- CORS Middleware ---
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
    model_path_env = os.getenv('MODEL_PATH_ENV')

    if model_path_env:
        model_path = model_path_env
        print(f"Server: Attempting to load model from environment variable path: {model_path}")
    else:
        model_path = "best.pt"
        if not os.path.exists(model_path):
            model_path = "weights/best.pt"
        print(f"Server: MODEL_PATH_ENV not set. Attempting to load model from default local path: {model_path}")

    try:
        model = YOLO(model_path)
        print(f"Server: Model loaded successfully from {model_path}. Ready for predictions!")
    except Exception as e:
        print(f"Server: Error loading model from {model_path}: {e}")
        import traceback
        traceback.print_exc()
        model = None

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

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    np_image = np.array(image)

    print(f"Server: Received image '{file.filename}' for detection.")

    results = model(np_image) # YOLO model expects numpy array or PIL Image

    detections = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            detection = {
                "box": boxes[i].tolist(),
                "score": float(scores[i]),
                "class_id": int(classes[i]),
                "class_name": model.names[int(classes[i])]
            }
            detections.append(detection)
    
    print(f"Server: Detected {len(detections)} objects.")
    return JSONResponse(content={"filename": file.filename, "detections": detections})


# --- WebSocket & Real-time Alert Endpoints ---

active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"WebSocket client connected: {websocket.client}")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received message from client: {data}")
    except Exception as e:
        print(f"WebSocket client disconnected or error: {e}")
    finally:
        active_connections.remove(websocket)
        print(f"WebSocket client disconnected: {websocket.client}")

async def broadcast_message(message: Dict):
    message_to_send = json.dumps(message)
    for connection in active_connections:
        try:
            await connection.send_text(message_to_send)
        except Exception as e:
            print(f"Error sending message to WebSocket client {connection.client}: {e}")

@app.post("/detection_alert/")
async def receive_detection_alert(detection_data: Dict):
    # This endpoint now expects 'alert_images' which is a list of URLs
    # It will simply pass this through to the WebSocket clients
    print(f"Received detection alert from Pi: {detection_data}")
    await broadcast_message(detection_data)
    return {"status": "alert received and broadcasted"}


# --- Local Development Server (Optional) ---
if __name__ == "__main__":
    import uvicorn
    os.environ['MODEL_PATH_ENV'] = 'weights/best.pt'
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import cv2
import numpy as np
from typing import List, Dict
import json
import asyncio
import datetime # Added for timestamp generation

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

# --- Live Feed Global Variables ---
# Store the latest annotated frame to send to new WebSocket connections
latest_annotated_frame_bytes: bytes = b''
# List of active WebSocket connections for the live feed
live_feed_connections: List[WebSocket] = []

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
        status_message = "YOLOv8 Object Detection API is running. Send POST requests with image files to `/detect_object/` to detect objects. WebSocket alerts available at `/ws`. Live video feed at `/live_feed_ws`. Send alerts from Pi to `/detection_alert/`."
    else:
        status_message = "YOLOv8 Object Detection API is running, but the model failed to load. Check server logs for details."
    return JSONResponse(content={"message": status_message})

# --- Object Detection Endpoint (Existing, unchanged, not directly used by current simulators) ---
@app.post("/detect_object/")
async def detect_object(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    np_image = np.array(image)

    print(f"Server: Received image '{file.filename}' for detection.")

    results = model(np_image)

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


# --- WebSocket for Alerts ---
active_alert_connections: List[WebSocket] = []

@app.websocket("/ws") # This is for text-based alerts
async def websocket_alert_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_alert_connections.append(websocket)
    print(f"WebSocket (Alerts) client connected: {websocket.client}")
    try:
        while True:
            # This WebSocket primarily receives text messages (if any) or stays open
            await websocket.receive_text()
    except WebSocketDisconnect:
        print(f"WebSocket (Alerts) client disconnected: {websocket.client}")
    except Exception as e:
        print(f"WebSocket (Alerts) error: {e}")
    finally:
        active_alert_connections.remove(websocket)

async def broadcast_alert_message(message: Dict):
    message_to_send = json.dumps(message)
    for connection in active_alert_connections:
        try:
            await connection.send_text(message_to_send)
        except Exception as e:
            print(f"Error sending alert message to WebSocket client {connection.client}: {e}")

@app.post("/detection_alert/")
async def receive_detection_alert(detection_data: Dict):
    print(f"Received detection alert from Pi: {detection_data}")
    await broadcast_alert_message(detection_data)
    return {"status": "alert received and broadcasted"}


# --- NEW: WebSocket for Live Video Stream with Detections ---
@app.websocket("/live_feed_ws")
async def live_feed_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    live_feed_connections.append(websocket)
    print(f"WebSocket (Live Feed) client connected: {websocket.client}")
    try:
        # Immediately send the latest frame if available to new connections
        if latest_annotated_frame_bytes:
            await websocket.send_bytes(latest_annotated_frame_bytes)

        # Keep the connection open indefinitely
        while True:
            # This WebSocket is for sending annotated frames.
            # We don't expect it to receive much from the dashboard,
            # but keep it alive.
            await asyncio.sleep(10) # Keep alive
    except WebSocketDisconnect:
        print(f"WebSocket (Live Feed) client disconnected: {websocket.client}")
    except Exception as e:
        print(f"WebSocket (Live Feed) error: {e}")
    finally:
        live_feed_connections.remove(websocket)


# --- NEW: Endpoint to receive frames from Raspberry Pi and process them ---
@app.post("/stream_frame/")
async def stream_frame(file: UploadFile = File(...)):
    global latest_annotated_frame_bytes
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    contents = await file.read()
    
    # Convert bytes to numpy array (OpenCV format)
    np_image = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Perform YOLO detection
    # Ensure 'person' and 'weapon' are correctly mapped from model.names
    results = model(frame, verbose=False) # verbose=False to suppress print statements

    # Prepare for drawing annotations and collecting alert data
    alert_detected = False
    weapon_confidence = 0.0
    detection_coordinates = [] # Stores [x1, y1, x2, y2] of the first detected weapon

    # Draw bounding boxes and labels
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            score = float(scores[i])
            class_id = int(classes[i])
            class_name = model.names[class_id]

            color = (0, 255, 0) # Default Green for general objects/persons
            if class_name == "weapon": # Assuming your model detects 'weapon' class
                color = (0, 0, 255) # Red for weapons
                
                # --- ALERT GENERATION LOGIC ---
                # Only trigger alert if confidence is above a threshold and we haven't alerted for this frame yet
                # You can adjust this threshold as needed
                if score > 0.5: # Example threshold
                    alert_detected = True 
                    weapon_confidence = score
                    detection_coordinates = [x1, y1, x2, y2]
                    # No need to break here, continue drawing all detections, but only send ONE alert per frame
                    
            elif class_name == "person": # Explicitly green for person if needed
                 color = (0, 255, 0) # Green for person

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {score:.2f}"
            # Ensure text fits within frame, adjust position if necessary
            text_x = x1
            text_y = y1 - 10
            if text_y < 10: # If text goes above frame
                text_y = y1 + 20
            
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- BROADCAST ALERT IF A WEAPON WAS DETECTED ---
    if alert_detected:
        # Encode the detected frame for the alert message
        # Use a higher quality for the alert image if desired
        ret_alert, buffer_alert = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        alert_image_base64 = None
        if ret_alert:
            alert_image_base64 = base64.b64encode(buffer_alert.tobytes()).decode('utf-8')

        alert_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "location": "Live Camera Feed Source", # Can be customized
            "detection_type": "weapon",
            "confidence": weapon_confidence,
            "coordinates": detection_coordinates,
            # Pass the image as a base64 data URI to the dashboard
            "image_url": f"data:image/jpeg;base64,{alert_image_base64}" if alert_image_base64 else None
        }
        await broadcast_alert_message(alert_data)
        print("ALERT: Weapon detected and alert broadcasted from live feed!") # For server logs

    # Encode the annotated frame back to JPEG bytes for the live feed display
    ret_live, buffer_live = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70]) # Use a different quality if desired
    if not ret_live:
        raise HTTPException(status_code=500, detail="Could not encode annotated image for live feed.")
    annotated_frame_bytes = buffer_live.tobytes()

    # Store the latest frame
    latest_annotated_frame_bytes = annotated_frame_bytes

    # Broadcast the annotated frame to all connected live feed WebSocket clients
    for connection in live_feed_connections:
        try:
            await connection.send_bytes(annotated_frame_bytes)
        except WebSocketDisconnect:
            # Client disconnected, will be removed by the websocket_endpoint handler
            pass
        except Exception as e:
            print(f"Error sending frame to live feed WebSocket client {connection.client}: {e}")

    return {"status": "Frame processed and broadcasted."}

# --- Local Development Server (Optional) ---
if __name__ == "__main__":
    import uvicorn
    # Make sure 'weights/best.pt' exists or is correctly configured
    os.environ['MODEL_PATH_ENV'] = 'weights/best.pt'
    uvicorn.run(app, host="0.0.0.0", port=8000)

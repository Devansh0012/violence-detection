import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import datetime
import motor.motor_asyncio
from passlib.context import CryptContext
from jose import jwt, JWTError
from typing import Optional
from fastapi.staticfiles import StaticFiles 
import os
import base64

# Constants
SECRET_KEY = "your-secret-key-here"  # Change this to a secure key
ALGORITHM = "HS256"
MONGO_URI = "mongodb+srv://devanshdubey0012:xkHHPuCdbmUYPcSm@violence.ktn5b.mongodb.net/?retryWrites=true&w=majority&appName=violence"  # Update with your MongoDB URI

# Create necessary directories
os.makedirs("detections", exist_ok=True)
os.makedirs("annotated_videos", exist_ok=True)

# Initialize object detection model
detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
detection_model.eval()

# Model Architecture for violence detection
class ViolenceDetectionModel(nn.Module):
    def __init__(self):
        super(ViolenceDetectionModel, self).__init__()
        self.features = models.mobilenet_v2(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 2),  # MUST match checkpoint's 2 outputs
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        return self.classifier(x)

# Initialize violence detection model
model = ViolenceDetectionModel()
model.load_state_dict(torch.load("retrained_model_v3.pth", map_location=torch.device('cpu')))
model.eval()

def detect_objects(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transforms.functional.to_tensor(rgb_frame)
    with torch.no_grad():
        predictions = detection_model([image_tensor])
    return predictions[0]

def draw_bounding_boxes(frame, detections):
    for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
        if score > 0.3:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    return frame

# Image preprocessing for violence detection
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MongoDB setup
try:
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    db = client.violence_detection
    users_collection = db.users
    detections_collection = db.detections
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    raise

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class UserBase(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Helper functions
def create_token(data: dict):
    try:
        to_encode = data.copy()
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        print(f"Token creation error: {str(e)}")
        raise

async def get_current_user(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401)
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def predict_violence(frame):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
            # Assuming index 1 is violence probability
            violence_score = output[0][1].item()
            print(f"Raw model output: {output}")
            print(f"Violence score: {violence_score}")
        return violence_score
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

# Routes
@app.post("/auth/signup")
async def signup(user: UserBase):
    try:
        existing_user = await users_collection.find_one({"username": user.username})
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        
        hashed_password = pwd_context.hash(user.password)
        await users_collection.insert_one({
            "username": user.username,
            "password": hashed_password
        })
        return {"message": "User created successfully"}
    except Exception as e:
        print(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login")
async def login(user: UserBase):
    try:
        db_user = await users_collection.find_one({"username": user.username})
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        if not pwd_context.verify(user.password, db_user["password"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        access_token = create_token({"sub": user.username})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "role": db_user.get("role", "user")
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        score = predict_violence(frame)
        if score is None:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        is_violent = score > 0.3
        timestamp = datetime.datetime.now().isoformat()
        
        if is_violent:
            detection_path = f"detections/{timestamp}.jpg"
            cv2.imwrite(detection_path, frame)
            await detections_collection.insert_one({
                "timestamp": timestamp,
                "score": score,
                "image_path": detection_path
            })
        
        return {"violence_detected": is_violent, "confidence_score": score, "timestamp": timestamp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     cap = cv2.VideoCapture("rtsp://your_camera_rtsp_url")
#     try:
#         while True:
#             data = await websocket.receive_bytes()
#             nparr = np.frombuffer(data, np.uint8)
#             frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if frame is None:
#                 continue
                
#             # Detect objects (people) in the frame
#             detections = detect_objects(frame)
            
#             # Violence prediction
#             score = predict_violence(frame)
#             is_violent = score > 0.2
            
#             # Draw bounding boxes - highlight violent areas in red
#             annotated_frame = frame.copy()
#             for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
#                 if score > 0.3:  # Only show confident detections
#                     x1, y1, x2, y2 = map(int, box)
#                     # Person is label 1 in COCO dataset
#                     if label == 1:  
#                         # Use red for violent scenes, green otherwise
#                         color = (0, 0, 255) if is_violent else (0, 255, 0)
#                         thickness = 2 if is_violent else 1
                        
#                         # Draw rectangle around person
#                         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        
#                         # Add confidence score text
#                         if is_violent:
#                             conf_text = f"Violence: {score:.2f}"
#                             cv2.putText(annotated_frame, conf_text, (x1, y1-10), 
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
#             # If violence detected, save the frame
#             if is_violent:
#                 timestamp = datetime.datetime.now().isoformat()
#                 detection_path = f"detections/{timestamp}.jpg"
#                 cv2.imwrite(detection_path, annotated_frame)
                
#                 # Save to database
#                 await detections_collection.insert_one({
#                     "timestamp": timestamp,
#                     "score": float(score),
#                     "image_path": detection_path
#                 })
            
#             # Encode the annotated frame as Base64 and send it back
#             _, jpeg = cv2.imencode('.jpg', annotated_frame)
#             import base64
#             encoded_img = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            
#             await websocket.send_json({
#                 "violence": is_violent,
#                 "score": float(score),
#                 "timestamp": datetime.datetime.now().isoformat(),
#                 "annotated_frame": encoded_img
#             })
#     except WebSocketDisconnect:
#         print("Client disconnected")
#     except Exception as e:
#         print(f"WebSocket error: {str(e)}")
#         await websocket.close()
# Replace the websocket_endpoint with RTSP processing
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture("rtsp://your_camera_rtsp_url")  # Replace with your RTSP URL
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detections = detect_objects(frame)
            annotated_frame = draw_bounding_boxes(frame.copy(), detections)
            score = predict_violence(frame)
            is_violent = score > 0.2
            
            # Save detection if violent
            if is_violent:
                timestamp = datetime.datetime.now().isoformat()
                detection_path = f"detections/{timestamp}.jpg"
                cv2.imwrite(detection_path, annotated_frame)
                await detections_collection.insert_one({
                    "timestamp": timestamp,
                    "score": float(score),
                    "image_path": detection_path
                })
            
            # Encode and send
            _, jpeg = cv2.imencode('.jpg', annotated_frame)
            encoded_img = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            await websocket.send_json({
                "violence": is_violent,
                "score": float(score),
                "timestamp": datetime.datetime.now().isoformat(),
                "annotated_frame": encoded_img
            })
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        cap.release()
        await websocket.close()

@app.post("/analyze_video")
async def analyze_video_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded video temporarily
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())
        
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Prepare video writer for annotated video
        timestamp = datetime.datetime.now().isoformat().replace(":", "-")
        annotated_video_path = f"annotated_videos/annotated_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
        
        # Process frames
        results = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detect_objects(frame)
            annotated_frame = draw_bounding_boxes(frame.copy(), detections)
            score = predict_violence(frame)
            is_violent = score > 0.2
            label = f"Violence: {'Yes' if is_violent else 'No'} {score:.2f}"
            cv2.putText(annotated_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if is_violent else (0, 255, 0), 2)
            out.write(annotated_frame)
            results.append({"violence": is_violent, "score": score})
        
        cap.release()
        out.release()
        os.remove(temp_video_path)
        
        return JSONResponse({
            "message": "Video analyzed successfully",
            "annotated_video": annotated_video_path,
            "results": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/annotated_videos", StaticFiles(directory="annotated_videos"), name="annotated_videos")
app.mount("/detections", StaticFiles(directory="detections"), name="detections")

@app.on_event("startup")
async def startup_event():
    try:
        await client.admin.command('ping')
        print("Successfully connected to MongoDB")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
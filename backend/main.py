import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
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
#from flask_cors import CORS

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
#CORS(app, resources={r"/*": {"origins": "*"}})
# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://violence-detection-tan.vercel.app"],  # Update with your frontend URL
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
    """Detect violence in a frame"""
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
        import traceback
        traceback.print_exc()
        return 0.0  # Return a safe default

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    rtsp_url = None
    use_rtsp = False
    cap = None
    
    try:
        # Process frames continuously
        while True:
            try:
                # First try to receive binary data (camera frames)
                data = await websocket.receive_bytes()
                
                # Process webcam frame from frontend
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Invalid image data received"})
                    continue
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                # If not binary, try to receive JSON (config)
                try:
                    json_data = await websocket.receive_json()
                    print(f"Received JSON config: {json_data}")
                    
                    # Configure RTSP if provided
                    if "rtsp_url" in json_data and json_data["rtsp_url"]:
                        rtsp_url = json_data["rtsp_url"]
                        use_rtsp = True
                        
                        # Close previous capture if any
                        if cap and cap.isOpened():
                            cap.release()
                            
                        # Open RTSP stream
                        print(f"Opening RTSP stream: {rtsp_url}")
                        cap = cv2.VideoCapture(rtsp_url)
                        
                        if not cap.isOpened():
                            await websocket.send_json({"error": f"Failed to open RTSP stream: {rtsp_url}"})
                            use_rtsp = False
                            continue
                        else:
                            await websocket.send_json({"message": f"RTSP stream opened: {rtsp_url}"})
                    elif "use_rtsp" in json_data and json_data["use_rtsp"] == False:
                        # User wants to use browser's webcam
                        use_rtsp = False
                        if cap and cap.isOpened():
                            cap.release()
                            cap = None
                        await websocket.send_json({"message": "Using browser webcam"})
                        continue
                    else:
                        # Use system camera as fallback
                        use_rtsp = True
                        
                        # Close previous capture if any
                        if cap and cap.isOpened():
                            cap.release()
                            
                        # Try to open system camera
                        print("Opening system camera")
                        cap = cv2.VideoCapture(0)  # 0 is usually the built-in webcam
                        
                        if not cap.isOpened():
                            await websocket.send_json({"error": "Failed to open system camera"})
                            use_rtsp = False
                            continue
                        else:
                            await websocket.send_json({"message": "System camera opened"})
                        
                except Exception as inner_e:
                    print(f"Error in websocket communication: {str(inner_e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Get frame from RTSP or system camera if configured
            if use_rtsp and cap and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    await websocket.send_json({"error": "Failed to read from stream"})
                    continue
            
            # Violence detection
            violence_score = predict_violence(frame)
            is_violent = violence_score > 0.3
            
            # Object detection
            detections = detect_objects(frame)
            
            # Create annotated frame with violence indicators
            annotated_frame = frame.copy()
            
            # Draw bounding boxes around detected people
            for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
                if score > 0.3:
                    x1, y1, x2, y2 = map(int, box)
                    # Person is label 1 in COCO dataset
                    if label == 1:
                        # Red for violence, green for non-violence
                        color = (0, 0, 255) if is_violent else (0, 255, 0)
                        thickness = 3 if is_violent else 1
                        
                        # Draw rectangle around person
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Add violence score text if violent
                        if is_violent:
                            text = f"Violence: {violence_score:.2f}"
                            cv2.putText(annotated_frame, text, (x1, y1-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add overall violence indicator at the top
            status_text = f"VIOLENCE DETECTED: {violence_score:.2f}" if is_violent else "No Violence"
            cv2.putText(annotated_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_violent else (0, 255, 0), 2)
            
            # Save violent frames
            if is_violent:
                timestamp = datetime.datetime.now().isoformat().replace(":", "-")
                detection_path = f"detections/{timestamp}.jpg"
                cv2.imwrite(detection_path, annotated_frame)
                
                # Log to database
                await detections_collection.insert_one({
                    "timestamp": timestamp,
                    "score": float(violence_score),
                    "image_path": detection_path
                })
            
            # Encode and send frame to frontend
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            
            try:
                await websocket.send_json({
                    "violence": is_violent,
                    "score": float(violence_score),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "annotated_frame": encoded_frame
                })
            except Exception as e:
                print(f"Send error: {str(e)}")
                break
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if cap and cap.isOpened():
            cap.release()
        await websocket.close()

# Replace your existing /analyze_video endpoint with the following:

@app.post("/analyze_video")
async def analyze_video_file(file: UploadFile = File(...)):
    try:
        # Save the uploaded video temporarily to disk
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare video writer for annotated video
        timestamp = datetime.datetime.now().isoformat().replace(":", "-")
        annotated_video_path = f"annotated_videos/annotated_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
        
        # Initialize summary statistics and samples
        frame_count = 0
        violence_frames = 0
        violent_segments = []
        current_segment = None
        sample_results = []
        sample_interval = max(1, total_frames // 50)  # sample at most 50 results
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            # Process every 5th frame for violence detection
            if frame_count % 5 == 0 or frame_count == 1:
                violence_score = predict_violence(frame)
                is_violent = violence_score > 0.3
                
                # Update violent segments information
                if is_violent:
                    violence_frames += 1
                    if current_segment is None:
                        current_segment = {
                            "start_frame": frame_count,
                            "start_time": frame_count / fps,
                            "scores": [violence_score]
                        }
                    else:
                        current_segment["scores"].append(violence_score)
                else:
                    if current_segment is not None:
                        current_segment["end_frame"] = frame_count - 5
                        current_segment["end_time"] = current_segment["end_frame"] / fps
                        current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
                        current_segment["avg_score"] = sum(current_segment["scores"]) / len(current_segment["scores"])
                        violent_segments.append(current_segment)
                        current_segment = None
                
                # Annotate frame with a simple status text
                annotated_frame = frame.copy()
                status_text = f"Violence: {violence_score:.2f}" if is_violent else "No Violence"
                cv2.putText(annotated_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255) if is_violent else (0, 255, 0), 2)
            else:
                annotated_frame = frame
            
            out.write(annotated_frame)
            
            # Optional sampling to reduce memory use
            if frame_count % sample_interval == 0:
                sample_results.append({
                    "frame": frame_count,
                    "time": frame_count / fps,
                    "violence_detected": is_violent if 'is_violent' in locals() else False,
                    "score": violence_score if 'violence_score' in locals() else 0.0
                })
        
        if current_segment is not None:
            current_segment["end_frame"] = frame_count
            current_segment["end_time"] = frame_count / fps
            current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
            current_segment["avg_score"] = sum(current_segment["scores"]) / len(current_segment["scores"])
            violent_segments.append(current_segment)
        
        cap.release()
        out.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        violence_percentage = (violence_frames / (frame_count / 5) * 100) if frame_count > 0 else 0
        if violence_percentage > 40:
            classification = "violent"
        elif violence_percentage > 10:
            classification = "ambiguous"
        else:
            classification = "non-violent"
        
        video_url = f"/annotated_videos/annotated_{timestamp}.mp4"
        
        # Assemble analysis document and store in MongoDB
        analysis_document = {
            "analysis_id": timestamp,
            "video_url": video_url,
            "summary": {
                "total_frames": frame_count,
                "violence_frames": violence_frames,
                "violence_percentage": violence_percentage,
                "classification": classification,
                "duration_seconds": frame_count / fps
            },
            "violent_segments": violent_segments,
            "sample_results": sample_results,
            "created_at": datetime.datetime.utcnow()
        }
        await db["analyses"].insert_one(analysis_document)
        
        return JSONResponse(analysis_document)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze_video/{analysis_id}")
async def get_video_analysis(analysis_id: str):
    """Get existing video analysis by ID"""
    try:
        analysis = await db["analyses"].find_one({"analysis_id": analysis_id})
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
            
        # Check for keyframes directory
        keyframes_dir = f"annotated_videos/keyframes_{analysis_id}"
        keyframes = []
        if os.path.exists(keyframes_dir):
            keyframes = [f"/annotated_videos/keyframes_{analysis_id}/{file}" for file in os.listdir(keyframes_dir)]
        
        analysis["keyframes"] = keyframes
        # Use jsonable_encoder to handle non-serializable types (like datetime)
        return JSONResponse(content=jsonable_encoder(analysis))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/ambiguous_samples")
async def get_ambiguous_samples():
    """Get images with ambiguous/uncertain violence classification"""
    try:
        # In a real app, you would query your database for images with confidence scores
        # in a specific range (e.g., 0.4-0.6) that need human verification
        
        # For now, return some mock data
        mock_samples = []
        
        # Fetch any entries from the detections collection with scores around 0.3-0.5
        cursor = detections_collection.find(
            {"score": {"$gte": 0.3, "$lte": 0.5}},
            {"_id": 0, "timestamp": 1, "score": 1, "image_path": 1}
        ).limit(5)
        
        samples = []
        async for doc in cursor:
            samples.append({
                "id": doc["timestamp"],  # Using timestamp as ID
                "image": f"/{doc['image_path']}",  # Path to the image
                "confidence": doc["score"]  # Confidence score
            })
            
        if not samples:
            # If no real data, provide mock data
            samples = [
                {
                    "id": 1,
                    "image": "/detections/mock_sample1.jpg",
                    "confidence": 0.45
                },
                {
                    "id": 2,
                    "image": "/detections/mock_sample2.jpg", 
                    "confidence": 0.51
                }
            ]
            
        return samples
    except Exception as e:
        print(f"Error getting ambiguous samples: {str(e)}")
        return []  # Return empty array on error

@app.post("/label_sample")
async def label_sample(data: dict):
    """Label a sample as violent or non-violent for model retraining"""
    try:
        sample_id = data.get("id")
        is_violent = data.get("is_violent")
        
        if sample_id is None or is_violent is None:
            raise HTTPException(status_code=400, detail="Missing id or is_violent in request")
            
        # In a real app, you would update your database and potentially add this to a dataset
        # for model retraining
        
        # For now, just log the action
        print(f"Sample {sample_id} labeled as {'violent' if is_violent else 'non-violent'}")
        
        # You could add to a "labeled_samples" collection for future model retraining
        await db["labeled_samples"].insert_one({
            "sample_id": sample_id,
            "is_violent": is_violent,
            "labeled_at": datetime.datetime.now()
        })
        
        return {"success": True, "message": "Sample labeled successfully"}
    except Exception as e:
        print(f"Error labeling sample: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_status")
async def model_status():
    """Check if the violence detection model is properly loaded"""
    try:
        # Create a random test tensor
        test_input = torch.rand(1, 3, 224, 224)
        
        # Run inference
        with torch.no_grad():
            output = model(test_input)
        
        return {
            "status": "ok", 
            "model_loaded": True,
            "output_shape": list(output.shape),
            "model_file": "retrained_model_v3.pth",
            "model_exists": os.path.exists("retrained_model_v3.pth"),
            "model_size_mb": round(os.path.getsize("retrained_model_v3.pth") / (1024 * 1024), 2) if os.path.exists("retrained_model_v3.pth") else None
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e),
            "model_file": "retrained_model_v3.pth",
            "model_exists": os.path.exists("retrained_model_v3.pth")
        }

app.mount("/annotated_videos", StaticFiles(directory="annotated_videos"), name="annotated_videos")
app.mount("/detections", StaticFiles(directory="detections"), name="detections")

@app.on_event("startup")
async def startup_event():
    try:
        # Test database connection
        info = await client.server_info()
        print("Successfully connected to MongoDB")

        print(f"Server will bind to: 0.0.0.0:{os.environ.get('PORT', 8000)}")
        
        # Verify model is loaded
        test_tensor = torch.zeros((1, 3, 224, 224))
        with torch.no_grad():
            test_output = model(test_tensor)
        print(f"Violence detection model loaded successfully. Output shape: {test_output.shape}")
        
        # Verify directories
        if os.path.exists("detections"):
            print(f"Detections directory exists: {len(os.listdir('detections'))} files")
        if os.path.exists("annotated_videos"):
            print(f"Annotated videos directory exists: {len(os.listdir('annotated_videos'))} files")
            
    except Exception as e:
        print(f"Startup error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 8000))  # Use Render's PORT
    )
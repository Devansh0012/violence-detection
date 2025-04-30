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
import json
from bson.objectid import ObjectId
import boto3
import asyncio
#from flask_cors import CORS

# Constants
SECRET_KEY = "your-secret-key-here"  # Change this to a secure key
ALGORITHM = "HS256"
MONGO_URI = "mongodb+srv://devanshdubey0012:xkHHPuCdbmUYPcSm@violence.ktn5b.mongodb.net/?retryWrites=true&w=majority&appName=violence"  # Update with your MongoDB URI

AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
S3_BUCKET = ""
S3_REGION = ""

# Create an S3 client
s3 = boto3.client(
    "s3",
    region_name=S3_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def generate_presigned_url(file_name):

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key = AWS_SECRET_ACCESS_KEY,
        region_name=S3_REGION,
    )

    try:
        presigned_url = s3_client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": file_name,
                # "ContentType": "image/jpeg",
                # "ACL": "public-read",
            },
            ExpiresIn=3600,  # URL expires in 1 hour
        )
        print(f"Generated path is {file_name}")
        print(f"Generated presigned URL: {presigned_url}")
        return presigned_url

    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None
    
def upload_file_to_s3(file_path: str, s3_key: str):
    try:
        print(f"Uploading {file_path} to S3 bucket {S3_BUCKET} with key {s3_key}")
        s3.upload_file(file_path, S3_BUCKET, s3_key)
        # Generate the correct URL format for the region
        file_url = generate_presigned_url(file_path)
        print(f"Upload successful: {file_url}")
        return file_url
    except Exception as exc:
        print(f"S3 upload error: {exc}")
        import traceback
        traceback.print_exc()
        
        # As a fallback, store the file locally and return a local URL
        print("Falling back to local storage")
        local_url = f"/annotated_videos/{os.path.basename(file_path)}"
        return local_url

# Create necessary directories
os.makedirs("detections", exist_ok=True)
os.makedirs("annotated_videos", exist_ok=True)

# Initialize object detection model
detection_model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
detection_model.eval()

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super(JSONEncoder, self).default(obj)

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://violence-detection-tan.vercel.app", "http://192.168.1.31:3000"],  # Update with your frontend URL
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

@app.post("/analyze_video")
async def analyze_video_file(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded video
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Open the video file
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a unique identifier based on timestamp
        timestamp = datetime.datetime.now().isoformat().replace(":", "-")
        local_annotated_path = f"annotated_videos/annotated_{timestamp}.mp4"
        
        # Create directory for keyframes
        keyframes_dir = f"annotated_videos/keyframes_{timestamp}"
        os.makedirs(keyframes_dir, exist_ok=True)
        
        # Set up video writer for annotated output
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(local_annotated_path, fourcc, fps, (width, height))
        
        # Initialize tracking variables
        frame_count = 0
        violence_frames = 0
        violent_segments = []
        current_segment = None
        sample_results = []
        keyframes = []
        sample_interval = max(1, total_frames // 50)  # Take ~50 samples throughout the video
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Only analyze every 5th frame to improve performance
            if frame_count % 5 == 0 or frame_count == 1:
                # Detect violence in the current frame
                violence_score = predict_violence(frame)
                is_violent = violence_score > 0.3
                
                # Track violent frames
                if is_violent:
                    violence_frames += 1
                    
                    # Start a new violent segment or continue the current one
                    if current_segment is None:
                        current_segment = {
                            "start_frame": frame_count,
                            "start_time": frame_count / fps,
                            "scores": [violence_score]
                        }
                        
                        # Save keyframe for the start of violent segment
                        keyframe_path = f"{keyframes_dir}/violent_start_{frame_count}.jpg"
                        cv2.imwrite(keyframe_path, frame)
                        
                        # Add to keyframes list (to be uploaded later)
                        keyframes.append({
                            "local_path": keyframe_path,
                            "s3_key": f"annotated_videos/keyframes_{timestamp}/violent_start_{frame_count}.jpg",
                            "type": "violent_start",
                            "frame": frame_count,
                            "time": frame_count / fps,
                            "score": violence_score
                        })
                    else:
                        current_segment["scores"].append(violence_score)
                        
                        # Periodically save keyframes for ongoing violent segments
                        if len(current_segment["scores"]) % 10 == 0:
                            keyframe_path = f"{keyframes_dir}/violent_ongoing_{frame_count}.jpg"
                            cv2.imwrite(keyframe_path, frame)
                            
                            # Add to keyframes list (to be uploaded later)
                            keyframes.append({
                                "local_path": keyframe_path,
                                "s3_key": f"annotated_videos/keyframes_{timestamp}/violent_ongoing_{frame_count}.jpg",
                                "type": "violent_ongoing",
                                "frame": frame_count,
                                "time": frame_count / fps,
                                "score": violence_score
                            })
                else:
                    # End the current violent segment if we have one
                    if current_segment is not None:
                        current_segment["end_frame"] = frame_count - 5
                        current_segment["end_time"] = current_segment["end_frame"] / fps
                        current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
                        current_segment["avg_score"] = sum(current_segment["scores"]) / len(current_segment["scores"])
                        violent_segments.append(current_segment)
                        
                        # Save keyframe for the end of violent segment
                        keyframe_path = f"{keyframes_dir}/violent_end_{frame_count}.jpg"
                        cv2.imwrite(keyframe_path, frame)
                        
                        # Add to keyframes list (to be uploaded later)
                        keyframes.append({
                            "local_path": keyframe_path,
                            "s3_key": f"annotated_videos/keyframes_{timestamp}/violent_end_{frame_count}.jpg",
                            "type": "violent_end",
                            "frame": frame_count,
                            "time": frame_count / fps,
                            "score": violence_score
                        })
                        
                        current_segment = None
                
                # Annotate the frame with violence information
                annotated_frame = frame.copy()
                status_text = f"Violence: {violence_score:.2f}" if is_violent else "No Violence"
                cv2.putText(annotated_frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255) if is_violent else (0, 255, 0), 2)
            else:
                # Use original frame if not analyzing
                annotated_frame = frame
            
            # Write the frame to output video
            out.write(annotated_frame)
            
            # Store sample results at regular intervals
            if frame_count % sample_interval == 0:
                # Also save these frames as regular keyframes
                keyframe_path = f"{keyframes_dir}/sample_{frame_count}.jpg"
                cv2.imwrite(keyframe_path, frame)
                
                # Add to keyframes list (to be uploaded later)
                keyframes.append({
                    "local_path": keyframe_path,
                    "s3_key": f"annotated_videos/keyframes_{timestamp}/sample_{frame_count}.jpg",
                    "type": "sample",
                    "frame": frame_count,
                    "time": frame_count / fps,
                    "score": violence_score if 'violence_score' in locals() else 0.0
                })
                
                sample_results.append({
                    "frame": frame_count,
                    "time": frame_count / fps,
                    "violence_detected": is_violent if 'is_violent' in locals() else False,
                    "score": violence_score if 'violence_score' in locals() else 0.0
                })
        
        # Close the last violent segment if video ends during a violent scene
        if current_segment is not None:
            current_segment["end_frame"] = frame_count
            current_segment["end_time"] = frame_count / fps
            current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
            current_segment["avg_score"] = sum(current_segment["scores"]) / len(current_segment["scores"])
            violent_segments.append(current_segment)
        
        # Clean up resources
        cap.release()
        out.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        # Calculate violence percentage and overall classification
        violence_percentage = (violence_frames / (frame_count / 5) * 100) if frame_count > 0 else 0
        if violence_percentage > 40:
            classification = "violent"
        elif violence_percentage > 10:
            classification = "ambiguous"
        else:
            classification = "non-violent"
        
        # Upload the annotated video to S3
        s3_key = f"annotated_videos/annotated_{timestamp}.mp4"
        video_url = upload_file_to_s3(local_annotated_path, s3_key)
        
        # Upload keyframes to S3 and collect their URLs
        keyframe_urls = []
        for kf in keyframes:
            try:
                kf_url = upload_file_to_s3(kf["local_path"], kf["s3_key"])
                keyframe_urls.append({
                    "url": kf_url,
                    "type": kf["type"],
                    "frame": kf["frame"],
                    "time": kf["time"],
                    "score": kf["score"]
                })
            except Exception as e:
                print(f"Failed to upload keyframe {kf['local_path']}: {e}")
        
        # Clean up local files
        os.remove(local_annotated_path)
        for kf in keyframes:
            try:
                os.remove(kf["local_path"])
            except Exception as e:
                print(f"Failed to remove keyframe {kf['local_path']}: {e}")
        os.rmdir(keyframes_dir)
        
        # Create analysis document to return and store in DB
        analysis_document = {
            "analysis_id": timestamp,
            "video_url": video_url,
            "keyframes": keyframe_urls,  # Store S3 URLs for keyframes
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
        
        # Store the analysis in the database
        result = await db["analyses"].insert_one(analysis_document)
        
        # Convert MongoDB ObjectId to string for JSON serialization
        analysis_document["_id"] = str(result.inserted_id)
        
        # Return the document as JSON
        return JSONResponse(content=json.loads(json.dumps(analysis_document, cls=JSONEncoder)))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connection attempt...")
    await websocket.accept()
    print("WebSocket connection accepted!")
    rtsp_url = None
    use_rtsp = False
    cap = None
    
    try:
        # Process frames continuously
        print("Entering WebSocket processing loop")
        while True:
            try:
                # Handle incoming messages - configuration or frames
                message = await websocket.receive()
                print(f"Received message type: {message.get('type')}")
                
                # Check if client disconnected
                if message.get("type") == "websocket.disconnect":
                    print("Client sent disconnect message")
                    break
                
                # Handle binary data (frames from browser webcam)
                if "bytes" in message:
                    # Only process webcam frames if we're not using RTSP
                    if not use_rtsp:
                        data = message["bytes"]
                        nparr = np.frombuffer(data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is None:
                            print("ERROR: Received invalid image data")
                            await websocket.send_json({"error": "Invalid image data received"})
                            continue
                    else:
                        # If using RTSP, ignore frames from browser
                        continue
                
                # Handle text data (configuration)
                elif "text" in message:
                    try:
                        json_data = json.loads(message["text"])
                        print(f"Received configuration: {json_data}")
                        
                        # Configure RTSP
                        if "rtsp_url" in json_data and json_data["rtsp_url"]:
                            rtsp_url = json_data["rtsp_url"]
                            
                            # Close previous capture if any
                            if cap and cap.isOpened():
                                cap.release()
                                cap = None
                            
                            print(f"Attempting to connect to RTSP: {rtsp_url}")
                            use_rtsp = True
                            cap = open_rtsp_stream_with_fallbacks(rtsp_url)
                            
                            if cap is None or not cap.isOpened():
                                print("Failed to open RTSP stream")
                                await websocket.send_json({"error": f"Failed to open RTSP stream: {rtsp_url}"})
                                use_rtsp = False
                            else:
                                print(f"Successfully opened RTSP stream: {rtsp_url}")
                                await websocket.send_json({"message": f"RTSP stream opened successfully"})
                        
                        # Handle stopping RTSP
                        if "use_rtsp" in json_data and not json_data["use_rtsp"]:
                            use_rtsp = False
                            if cap and cap.isOpened():
                                cap.release()
                                cap = None
                            await websocket.send_json({"message": "Switched to webcam mode"})
                                
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        await websocket.send_json({"error": f"Invalid JSON: {str(e)}"})
                        continue
                
                # Process RTSP frames if available
                if use_rtsp and cap and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to read frame from RTSP stream")
                        # Try to reconnect to RTSP stream
                        cap.release()
                        cap = open_rtsp_stream_with_fallbacks(rtsp_url)
                        if cap is None or not cap.isOpened():
                            await websocket.send_json({"error": "Failed to reconnect to RTSP stream"})
                            use_rtsp = False
                        continue
                
                # Skip if no frame is available to process
                if (use_rtsp and ("frame" not in locals() or frame is None)) or \
                   (not use_rtsp and ("frame" not in locals() or frame is None)):
                    # No frame to process
                    if use_rtsp:
                        # For RTSP we continue trying to get frames
                        continue
                    else:
                        # For webcam we just wait for more data
                        await asyncio.sleep(0.01)
                        continue
                
                # Process the frame with violence detection model
                violence_score = predict_violence(frame)
                is_violent = violence_score > 0.3
                
                # Object detection
                detections = detect_objects(frame)
                
                # Create annotated frame
                annotated_frame = frame.copy()
                
                # Draw bounding boxes around detected people
                for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
                    if score > 0.3:
                        x1, y1, x2, y2 = map(int, box)
                        # Person is label 1 in COCO dataset
                        if label == 1:
                            color = (0, 0, 255) if is_violent else (0, 255, 0)
                            thickness = 3 if is_violent else 1
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Add status text
                status_text = f"Violence: {violence_score:.2f}" if is_violent else "No Violence"
                cv2.putText(annotated_frame, status_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (0, 0, 255) if is_violent else (0, 255, 0), 2)
                
                # Convert frame to JPEG and send it
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                jpg_bytes = buffer.tobytes()
                await websocket.send_bytes(jpg_bytes)
                
                # Send analysis data as JSON
                await websocket.send_json({
                    "is_violent": is_violent,
                    "violence_score": violence_score,
                    "timestamp": datetime.datetime.now().isoformat()
                })
                
                # Add a small delay to control frame rate
                if use_rtsp:
                    await asyncio.sleep(0.03)  # ~30fps for RTSP
                
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                break
                
            except Exception as e:
                print(f"Error processing websocket message: {str(e)}")
                import traceback
                traceback.print_exc()
                # Don't break the loop, try to continue processing
                await asyncio.sleep(0.5)  # Add delay before retry
                continue
                
    except WebSocketDisconnect:
        print("WebSocketDisconnect caught in outer try block")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up resources
        if cap and cap.isOpened():
            cap.release()
        print("WebSocket connection closed")

# Add this function to your main.py file
def open_rtsp_stream_with_fallbacks(rtsp_url):
    """Try multiple methods to open RTSP stream and return the first working one"""
    print(f"Attempting to open RTSP stream: {rtsp_url}")
    
    # Method 1: Standard OpenCV
    print("Trying standard OpenCV connection...")
    cap = cv2.VideoCapture(rtsp_url)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print("Standard OpenCV connection successful")
            return cap
        cap.release()
    
    # Method 2: FFMPEG with TCP
    print("Trying FFMPEG with TCP transport...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print("FFMPEG with TCP transport successful")
            return cap
        cap.release()
    
    # Method 3: FFMPEG with UDP
    print("Trying FFMPEG with UDP transport...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print("FFMPEG with UDP transport successful")
            return cap
        cap.release()
    
    # Method 4: FFMPEG with additional options
    print("Trying FFMPEG with additional options...")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000|stimeout;5000000|max_delay;500000"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print("FFMPEG with additional options successful")
            return cap
        cap.release()
    
    # Method 5: GStreamer pipeline if available
    if 'GStreamer' in cv2.getBuildInformation():
        print("Trying GStreamer pipeline...")
        gst_str = (f'rtspsrc location={rtsp_url} latency=0 ! '
                  'rtph264depay ! h264parse ! avdec_h264 ! '
                  'videoconvert ! appsink')
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print("GStreamer pipeline successful")
                return cap
            cap.release()
    
    print("All RTSP connection methods failed")
    return None

@app.get("/analyze_video/{analysis_id}")
async def get_video_analysis(analysis_id: str):
    """Get existing video analysis by ID"""
    try:
        analysis = await db["analyses"].find_one({"analysis_id": analysis_id})
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Keyframes should now be in the database document directly
        # We don't need to check for local keyframes anymore
        
        # Convert MongoDB ObjectId to string
        if "_id" in analysis:
            analysis["_id"] = str(analysis["_id"])
        
        # Convert other datetime fields to ISO format
        if "created_at" in analysis:
            analysis["created_at"] = analysis["created_at"].isoformat()
            
        # Return using the custom JSONEncoder
        return JSONResponse(content=json.loads(json.dumps(analysis, cls=JSONEncoder)))
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
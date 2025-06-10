import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import datetime
import traceback
import motor.motor_asyncio
import os
import json
import asyncio
import threading
import queue
import time
from collections import deque
import uvicorn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import jwt, JWTError
from typing import Optional
from fastapi.staticfiles import StaticFiles 
import base64
from bson.objectid import ObjectId
import boto3
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import euclidean
import mediapipe as mp

# Constants
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
MONGO_URI = "mongodb+srv://devanshdubey0012:xkHHPuCdbmUYPcSm@violence.ktn5b.mongodb.net/?retryWrites=true&w=majority&appName=violence"

# AWS credentials
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
            },
            ExpiresIn=3600,
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
        file_url = generate_presigned_url(file_path)
        print(f"Upload successful: {file_url}")
        return file_url
    except Exception as exc:
        print(f"S3 upload error: {exc}")
        traceback.print_exc()
        
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
        import numpy as np
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (np.float_, np.float32, np.float64)):  # Add numpy float types
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(JSONEncoder, self).default(obj)

# Model Architecture for violence detection
class ViolenceDetectionModel(nn.Module):
    def __init__(self):
        super(ViolenceDetectionModel, self).__init__()
        self.features = models.mobilenet_v2(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280, 2),
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

# Threading-based frame reader for RTSP streams
class RTSPFrameReader:
    def __init__(self, rtsp_url, buffer_size=2):
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.cap = None
        self.reading = False
        self.thread = None
        
    def start(self):
        if self.reading:
            return False
            
        self.cap = self._create_capture()
        if not self.cap or not self.cap.isOpened():
            print("Failed to open RTSP stream")
            return False
            
        self.reading = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        return True
        
    def _create_capture(self):
        """Create OpenCV capture with optimized settings for RTSP"""
        print(f"Creating RTSP capture for: {self.rtsp_url}")
        
        # Try different methods in order of preference
        methods = [
            lambda: self._try_standard_capture(),
            lambda: self._try_ffmpeg_tcp(),
            lambda: self._try_ffmpeg_udp(),
            lambda: self._try_ffmpeg_advanced(),
        ]
        
        for method in methods:
            cap = method()
            if cap and cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print("Successfully created RTSP capture")
                    return cap
                cap.release()
        
        print("All RTSP capture methods failed")
        return None
        
    def _try_standard_capture(self):
        print("Trying standard OpenCV capture...")
        cap = cv2.VideoCapture(self.rtsp_url)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
        return cap
        
    def _try_ffmpeg_tcp(self):
        print("Trying FFMPEG with TCP...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1024000"
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
        
    def _try_ffmpeg_udp(self):
        print("Trying FFMPEG with UDP...")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;1024000"
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
        
    def _try_ffmpeg_advanced(self):
        print("Trying FFMPEG with advanced options...")
        options = (
            "rtsp_transport;tcp|"           # Use TCP for more reliable streaming
            "buffer_size;0|"               # Minimize buffering
            "max_delay;0|"                 # No frame delay
            "reorder_queue_size;0|"        # Disable reordering
            "stimeout;0|"                  # Minimal timeout
            "fflags;nobuffer+discardcorrupt|"  # Disable input buffering and discard corrupt frames
            "flags;low_delay|"             # Enable low delay mode
            "probesize;32|"               # Reduce probe size
            "analyzeduration;0|"           # Reduce analysis duration
            "sync;ext"                     # Use external synchronization
        )
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer size
        return cap
        
    def _read_frames(self):
        """Continuously read frames in background thread"""
        consecutive_failures = 0
        max_failures = 10

        while self.reading:
            try:
                if not self.cap or not self.cap.isOpened():
                    print("Reconnecting to RTSP stream...")
                    if self.cap:
                        self.cap.release()
                    self.cap = self._create_capture()
                    if not self.cap:
                        time.sleep(2)
                        continue
                    
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    consecutive_failures = 0

                    # Clear old frames to maintain low latency (keep only 1 frame)
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                        
                    # Add new frame
                    try:
                        self.frame_queue.put(frame, timeout=0.01)  # Very short timeout
                    except queue.Full:
                        # If queue is full, remove old frame and add new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame, timeout=0.01)
                        except queue.Empty:
                            pass
                        
                else:
                    consecutive_failures += 1
                    print(f"Failed to read frame (attempt {consecutive_failures})")
                    if consecutive_failures >= max_failures:
                        print(f"Too many consecutive failures ({consecutive_failures}), reconnecting...")
                        consecutive_failures = 0
                        if self.cap:
                            self.cap.release()
                            self.cap = None
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error in frame reading thread: {e}")
                consecutive_failures += 1
                time.sleep(0.5)
                
    def get_frame(self):
        """Get the latest frame from the queue"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
            
    def stop(self):
        """Stop the frame reader"""
        self.reading = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        # Clear the queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

def detect_objects(frame):
    """Detect objects in frame and return predictions"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transforms.functional.to_tensor(rgb_frame)
    with torch.no_grad():
        predictions = detection_model([image_tensor])
    return predictions[0]

def detect_people_in_frame(frame, confidence_threshold=0.6):
    """Detect people in the frame and return their count and bounding boxes"""
    detections = detect_objects(frame)
    person_count = 0
    person_boxes = []
    
    for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
        # Person class in COCO dataset is label 1
        if label == 1 and score > confidence_threshold:
            person_count += 1
            person_boxes.append({
                'box': box.tolist(),
                'score': score.item(),
                'label': label.item()
            })
    
    return person_count, person_boxes

def calculate_motion_intensity(frame, prev_frame):
    """Calculate motion with better normal motion filtering"""
    try:
        if prev_frame is None:
            return 0.0
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Quick check for identical frames
        if np.array_equal(gray1, gray2):
            return 0.0
        
        # Method 1: Frame difference with higher threshold for normal motion
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)  # Increased threshold
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = gray1.shape[0] * gray1.shape[1]
        motion_percentage = (motion_pixels / total_pixels) * 100
        
        # Method 2: Optical flow with better parameters
        features = cv2.goodFeaturesToTrack(
            gray1,
            maxCorners=150,     # Reduced from 200
            qualityLevel=0.3,   # Increased from 0.2
            minDistance=7,      # Increased from 5
            blockSize=7
        )
        
        optical_flow_motion = 0.0
        if features is not None and len(features) > 0:
            next_features, status, error = cv2.calcOpticalFlowPyrLK(
                gray1, gray2, features, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            good_new = next_features[status == 1]
            good_old = features[status == 1]
            
            if len(good_new) > 0 and len(good_old) > 0:
                motion_vectors = good_new - good_old
                magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
                
                if len(magnitudes) > 0:
                    # More aggressive outlier removal for normal motion
                    mean_magnitude = np.mean(magnitudes)
                    std_magnitude = np.std(magnitudes)
                    valid_magnitudes = magnitudes[magnitudes < mean_magnitude + 2 * std_magnitude]
                    
                    if len(valid_magnitudes) > 0:
                        optical_flow_motion = np.mean(valid_magnitudes)
                        
                        # Don't amplify normal motion as much
                        if optical_flow_motion < 10.0:  # Normal motion range
                            optical_flow_motion *= 1.2
                        else:  # Potentially violent motion
                            optical_flow_motion *= 1.8
        
        # Combine methods with less aggressive amplification
        final_motion = (motion_percentage * 0.7 + optical_flow_motion * 0.3)
        
        # Less aggressive overall amplification
        if final_motion < 10.0:  # Normal motion
            final_motion = min(final_motion * 1.1, 100.0)
        else:  # High motion
            final_motion = min(final_motion * 1.3, 100.0)
        
        print(f"Motion: pixels={motion_percentage:.2f}%, optical={optical_flow_motion:.2f}, final={final_motion:.2f}")
        
        return final_motion
        
    except Exception as e:
        print(f"Error calculating motion intensity: {str(e)}")
        return 0.0

def analyze_pose_changes(person_boxes, prev_person_boxes):
    """Analyze changes in person poses/positions"""
    if not person_boxes or not prev_person_boxes:
        return 0.0
    
    # Simple analysis of bounding box changes
    position_changes = []
    
    for box in person_boxes:
        current_center = [(box['box'][0] + box['box'][2])/2, (box['box'][1] + box['box'][3])/2]
        
        # Find closest previous box
        min_distance = float('inf')
        for prev_box in prev_person_boxes:
            prev_center = [(prev_box['box'][0] + prev_box['box'][2])/2, 
                          (prev_box['box'][1] + prev_box['box'][3])/2]
            distance = np.sqrt((current_center[0] - prev_center[0])**2 + 
                             (current_center[1] - prev_center[1])**2)
            min_distance = min(min_distance, distance)
        
        position_changes.append(min_distance)
    
    return np.mean(position_changes) if position_changes else 0.0

# Enhanced Violence Detection System
class AdvancedViolenceDetector:
    def __init__(self):
        # Simplified initialization without complex MediaPipe setup
        self.violence_patterns = deque(maxlen=30)
        self.pose_history = deque(maxlen=15)
        self.motion_vectors = deque(maxlen=20)
        self.crowd_dynamics = deque(maxlen=10)
        
        # Simplified feature weights
        self.feature_weights = {
            'cnn_score': 0.5,       # Increased CNN weight
            'motion_intensity': 0.3,
            'pose_violence': 0.2,
        }

    def extract_pose_features(self, frame):
        """Simplified pose feature extraction"""
        return {
            'aggressive_poses': 0,
            'rapid_movements': 0,
            'fighting_stances': 0,
            'arm_extensions': 0
        }

    def analyze_crowd_behavior(self, person_boxes, frame_shape):
        """Simplified crowd behavior analysis"""
        if len(person_boxes) < 2:
            return {'crowd_violence': 0, 'density': 0, 'clustering': 0}
        
        # Simple density calculation
        frame_area = frame_shape[0] * frame_shape[1]
        person_area = sum([(box['box'][2] - box['box'][0]) * (box['box'][3] - box['box'][1]) 
                          for box in person_boxes])
        density = min(person_area / frame_area, 1.0)
        
        # Simple clustering based on number of people
        clustering = min(len(person_boxes) / 5.0, 1.0)
        
        # Violence score based on density and clustering
        violence_score = 0
        if density > 0.3 and len(person_boxes) > 3:
            violence_score = 0.3
        
        return {
            'crowd_violence': violence_score,
            'density': density,
            'clustering': clustering,
            'movement': 0
        }

    def enhanced_optical_flow_analysis(self, frame, prev_frame):
        """Simplified optical flow analysis"""
        return {'flow_magnitude': 0, 'violence_patterns': 0, 'chaos_index': 0}

    def spatial_temporal_analysis(self):
        """Simplified spatial-temporal analysis"""
        return 0.0

# Violence detection with improved logic
class ViolenceAnalyzer:
    def __init__(self, history_size=8):  # Reduced history size
        self.score_history = deque(maxlen=history_size)
        self.motion_history = deque(maxlen=history_size)
        self.prev_frame = None
        self.prev_person_boxes = []
        self.frame_count = 0
        self.frame_skip = 2
        
        # Better balanced thresholds
        self.violence_threshold = 0.6    # Increased from 0.4
        self.motion_threshold = 15.0     # Increased from 5.0
        self.ensemble_threshold = 0.5    # Increased from 0.35
        self.cnn_confidence_threshold = 0.7  # High threshold for CNN alone
        
        # Initialize MediaPipe with error handling
        try:
            self.pose_detector = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            self.mp_pose = mp.solutions.pose
            self.use_pose = True
            print("MediaPipe pose detection initialized successfully")
        except Exception as e:
            print(f"MediaPipe initialization failed: {e}")
            self.pose_detector = None
            self.use_pose = False

    def _get_model_prediction(self, frame):
        """Get CNN model prediction"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb)
            input_batch = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = model(input_batch)
                violence_score = output[0][1].item()
                
            return violence_score
        except Exception as e:
            print(f"Model prediction error: {str(e)}")
            return 0.0

    def _calculate_motion_intensity(self, frame):
        """Calculate motion intensity with better normal motion filtering"""
        if self.prev_frame is None:
            return 0.0
        
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Frame difference with higher threshold to ignore small movements
            diff = cv2.absdiff(gray1, gray2)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)  # Higher threshold
            motion_pixels = cv2.countNonZero(thresh)
            total_pixels = gray1.shape[0] * gray1.shape[1]
            motion_percentage = (motion_pixels / total_pixels) * 100
            
            # Optical flow for more accurate motion
            features = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            flow_magnitude = 0
            if features is not None and len(features) > 5:
                next_features, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, features, None)
                good_new = next_features[status == 1]
                good_old = features[status == 1]
                
                if len(good_new) > 0:
                    motion_vectors = good_new - good_old
                    magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
                    flow_magnitude = np.mean(magnitudes)
            
            # Combine both methods - less aggressive amplification
            final_motion = (motion_percentage * 0.6 + flow_magnitude * 0.4)
            
            # Only amplify if motion is significant
            if final_motion > 10.0:
                final_motion *= 1.2  # Reduced amplification
            
            return final_motion
            
        except Exception as e:
            print(f"Error calculating motion: {e}")
            return 0.0

    def _extract_pose_features(self, frame):
        """Extract pose features with stricter criteria"""
        if not self.use_pose or self.pose_detector is None:
            return 0.0
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Stricter violence indicators
                violence_score = 0
                
                # Check for raised arms (fighting pose) - stricter criteria
                left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                
                # Both arms must be significantly raised
                left_arm_raised = left_wrist.y < (left_shoulder.y - 0.1)
                right_arm_raised = right_wrist.y < (right_shoulder.y - 0.1)
                
                if left_arm_raised and right_arm_raised:
                    violence_score += 0.4  # Both arms raised
                elif left_arm_raised or right_arm_raised:
                    violence_score += 0.2  # One arm raised
                
                # Wide stance (fighting stance) - stricter criteria
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                hip_distance = abs(left_hip.x - right_hip.x)
                
                if hip_distance > 0.4:  # Very wide stance
                    violence_score += 0.3
                elif hip_distance > 0.3:  # Wide stance
                    violence_score += 0.1
                
                return min(violence_score, 1.0)
            
            return 0.0
            
        except Exception as e:
            print(f"Pose extraction error: {e}")
            return 0.0

    def analyze_frame(self, frame):
        """Improved frame analysis with better violence discrimination"""
        try:
            self.frame_count += 1
            
            # Get person detection
            person_count, person_boxes = detect_people_in_frame(frame, confidence_threshold=0.6)
            
            if person_count == 0:
                return 0.0, 0, False, {'reason': 'no_people'}
            
            # 1. Get CNN model prediction (primary indicator)
            cnn_score = self._get_model_prediction(frame)
            
            # 2. Calculate motion intensity
            motion_intensity = 0.0
            if self.frame_count % self.frame_skip == 0:
                if self.prev_frame is not None:
                    motion_intensity = self._calculate_motion_intensity(frame)
                    self.motion_history.append(motion_intensity)
                self.prev_frame = frame.copy()
            else:
                motion_intensity = self.motion_history[-1] if self.motion_history else 0.0
            
            # 3. Extract pose features
            pose_score = self._extract_pose_features(frame)
            
            # 4. Better ensemble logic with stricter criteria
            ensemble_score = 0.0
            
            # Primary CNN-based decision
            if cnn_score > 0.8:  # Very high CNN confidence
                ensemble_score = cnn_score * 0.9 + pose_score * 0.1
            elif cnn_score > 0.6:  # High CNN confidence
                # Consider motion only if it's significant
                motion_factor = min(motion_intensity / 25.0, 1.0) if motion_intensity > 10.0 else 0.0
                ensemble_score = cnn_score * 0.7 + pose_score * 0.2 + motion_factor * 0.1
            elif cnn_score > 0.4:  # Medium CNN confidence
                # Require both motion and pose indicators
                motion_factor = min(motion_intensity / 20.0, 1.0) if motion_intensity > 15.0 else 0.0
                if motion_factor > 0.3 and pose_score > 0.3:  # Both must be present
                    ensemble_score = cnn_score * 0.5 + pose_score * 0.3 + motion_factor * 0.2
                else:
                    ensemble_score = cnn_score * 0.3  # Heavily discount without supporting evidence
            else:  # Low CNN confidence
                # Very strict requirements
                motion_factor = min(motion_intensity / 30.0, 1.0) if motion_intensity > 20.0 else 0.0
                if motion_factor > 0.5 and pose_score > 0.5:  # Both must be high
                    ensemble_score = cnn_score * 0.3 + pose_score * 0.4 + motion_factor * 0.3
                else:
                    ensemble_score = 0.0  # Reject low confidence without strong supporting evidence
            
            # Apply person count penalty for single person
            if person_count == 1 and ensemble_score > 0:
                ensemble_score *= 0.7  # Reduce score for single person scenarios
            
            # Update history
            self.score_history.append(ensemble_score)
            
            # Temporal smoothing with conservative approach
            if len(self.score_history) >= 3:
                recent_scores = list(self.score_history)[-5:]
                # More conservative weights - less influence from single high scores
                if len(recent_scores) >= 4:
                    weights = [0.1, 0.2, 0.3, 0.4]
                    temporal_score = sum(score * weight for score, weight in zip(recent_scores[-4:], weights))
                else:
                    temporal_score = np.mean(recent_scores) * 0.8  # Discount factor
            else:
                temporal_score = ensemble_score * 0.6  # Heavy discount for insufficient history
            
            # Make final decision with multiple strict criteria
            is_violent = False
            confidence = 0.0
            
            # Multiple validation criteria - ALL must be met for violence detection
            violence_criteria = {
                'temporal_high': temporal_score > self.ensemble_threshold,
                'cnn_confident': cnn_score > self.cnn_confidence_threshold,
                'motion_significant': motion_intensity > self.motion_threshold,
                'sustained_detection': False
            }
            
            # Check for sustained detection
            if len(self.score_history) >= 4:
                high_scores = sum(1 for s in list(self.score_history)[-4:] if s > 0.4)
                violence_criteria['sustained_detection'] = high_scores >= 3
            
            # Decision logic - require multiple criteria
            if violence_criteria['cnn_confident'] and violence_criteria['motion_significant']:
                # High CNN + High Motion
                is_violent = True
                confidence = max(temporal_score, cnn_score)
            elif violence_criteria['temporal_high'] and violence_criteria['sustained_detection']:
                # Sustained high temporal score
                is_violent = True
                confidence = temporal_score
            elif (cnn_score > 0.85 and motion_intensity > 10.0 and pose_score > 0.3):
                # Very high CNN with some supporting evidence
                is_violent = True
                confidence = cnn_score
            else:
                # Default to non-violent
                is_violent = False
                confidence = 0.0
            
            # Additional safety checks
            if person_count == 1 and motion_intensity < 20.0:
                # Single person with low motion - likely not violence
                is_violent = False
                confidence = 0.0
            
            if cnn_score < 0.3:
                # Very low CNN score - override everything
                is_violent = False
                confidence = 0.0
            
            # Update previous person boxes
            self.prev_person_boxes = person_boxes
            
            analysis_data = {
                'cnn_score': float(cnn_score),
                'ensemble_score': float(ensemble_score),
                'temporal_score': float(temporal_score),
                'confidence': float(confidence),
                'motion_intensity': float(motion_intensity),
                'pose_score': float(pose_score),
                'person_count': int(person_count),
                'violence_criteria': violence_criteria,
                'decision_factors': {
                    'cnn_high': cnn_score > self.cnn_confidence_threshold,
                    'motion_high': motion_intensity > self.motion_threshold,
                    'temporal_threshold': temporal_score > self.ensemble_threshold,
                    'sustained': violence_criteria['sustained_detection']
                }
            }
            
            print(f"Analysis: CNN={cnn_score:.3f}, Motion={motion_intensity:.1f}, Pose={pose_score:.3f}, "
                  f"Ensemble={ensemble_score:.3f}, Temporal={temporal_score:.3f}, Violent={is_violent}")
            
            return confidence, person_count, is_violent, analysis_data
            
        except Exception as e:
            print(f"Error in violence analysis: {str(e)}")
            traceback.print_exc()
            return 0.0, 0, False, {}

# Initialize violence analyzer
violence_analyzer = ViolenceAnalyzer()

# Add the missing predict_violence function for backward compatibility
def predict_violence(frame):
    """Legacy function that uses the enhanced ViolenceAnalyzer"""
    violence_score, person_count, is_violent, analysis_data = violence_analyzer.analyze_frame(frame)
    return violence_score

def draw_bounding_boxes(frame, detections, is_violent=False, analysis_data=None):
    """Simplified bounding box drawing"""
    for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
        if score > 0.5 and label == 1:  # Person detection
            x1, y1, x2, y2 = map(int, box)
            
            # Simple color coding
            if is_violent:
                color = (0, 0, 255)  # Red for violent
                thickness = 3
                label_text = "VIOLENCE DETECTED"
            else:
                color = (0, 255, 0)  # Green for safe
                thickness = 2
                label_text = "SAFE"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add label
            cv2.putText(frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add status overlay
    if analysis_data:
        status = "VIOLENCE DETECTED!" if is_violent else "SAFE"
        status_color = (0, 0, 255) if is_violent else (0, 255, 0)
        
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        
        # Add confidence info
        confidence = analysis_data.get('confidence', 0)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add CNN score
        cnn_score = analysis_data.get('cnn_score', 0)
        cv2.putText(frame, f"CNN Score: {cnn_score:.2f}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://violence-detection-tan.vercel.app", "http://192.168.1.31:3000"],
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

# WebSocket endpoint with improved RTSP handling
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connection attempt...")
    await websocket.accept()
    print("WebSocket connection accepted!")
    
    rtsp_reader = None
    use_rtsp = False
    frame_count = 0
    last_frame_time = time.time()
    
    try:
        print("Entering WebSocket processing loop")
        while True:
            try:
                # Check for new messages without blocking
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=0.01)
                    print(f"Received message type: {message.get('type')}")
                    
                    if message.get("type") == "websocket.disconnect":
                        print("Client sent disconnect message")
                        break
                    
                    # Handle binary data (frames from browser webcam)
                    if "bytes" in message:
                        if not use_rtsp:
                            data = message["bytes"]
                            nparr = np.frombuffer(data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is None:
                                print("ERROR: Received invalid image data")
                                await websocket.send_json({"error": "Invalid image data received"})
                                continue
                        else:
                            continue
                    
                    # Handle text data (configuration)
                    elif "text" in message:
                        try:
                            json_data = json.loads(message["text"])
                            print(f"Received configuration: {json_data}")
                            
                            # Configure RTSP
                            if "rtsp_url" in json_data and json_data["rtsp_url"]:
                                rtsp_url = json_data["rtsp_url"]
                                
                                # Stop existing reader
                                if rtsp_reader:
                                    rtsp_reader.stop()
                                    await asyncio.sleep(0.1)  # Give time for cleanup
                                    rtsp_reader = None
                                
                                print(f"Starting RTSP reader for: {rtsp_url}")
                                use_rtsp = True
                                rtsp_reader = RTSPFrameReader(rtsp_url, buffer_size=1)  # Reduced buffer
                                
                                if rtsp_reader.start():
                                    await websocket.send_json({"message": "RTSP stream started successfully"})
                                else:
                                    await websocket.send_json({"error": f"Failed to start RTSP stream: {rtsp_url}"})
                                    use_rtsp = False
                                    rtsp_reader = None
                            
                            # Handle stopping RTSP
                            if "use_rtsp" in json_data and not json_data["use_rtsp"]:
                                use_rtsp = False
                                if rtsp_reader:
                                    rtsp_reader.stop()
                                    rtsp_reader = None
                                await websocket.send_json({"message": "Switched to webcam mode"})
                                    
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            await websocket.send_json({"error": f"Invalid JSON: {str(e)}"})
                            continue
                
                except asyncio.TimeoutError:
                    pass  # No new messages, continue with frame processing
                except Exception as e:
                    print(f"Error receiving message: {str(e)}")
                    
                # Get frame for processing
                frame = None
                if use_rtsp and rtsp_reader:
                    frame = rtsp_reader.get_frame()
                    if frame is None:
                        await asyncio.sleep(0.001)  # Minimal sleep
                        continue
                elif "frame" in locals() and frame is not None:
                    pass
                else:
                    await asyncio.sleep(0.001)
                    continue

                if frame is None:
                    continue

                # Process frame with minimal overhead
                violence_score, person_count, is_violent, analysis_data = violence_analyzer.analyze_frame(frame)
                
                # Create annotated frame with minimal processing
                detections = detect_objects(frame)
                annotated_frame = draw_bounding_boxes(frame, detections, is_violent, analysis_data)
                
                # Optimize JPEG encoding
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]  # Reduced quality for less latency
                _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
                
                try:
                    # Send frame and analysis data
                    await websocket.send_bytes(buffer.tobytes())
                    await websocket.send_json({
                        "is_violent": is_violent,
                        "violence_score": violence_score,
                        "person_count": person_count,
                        "has_people": person_count > 0,
                        "analysis_data": analysis_data,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Error sending frame: {str(e)}")
                    break

                # Minimal delay between frames
                await asyncio.sleep(0.001)

            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Error processing websocket message: {str(e)}")
                await asyncio.sleep(0.1)
                continue
                
    finally:
        if rtsp_reader:
            rtsp_reader.stop()

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
        sample_interval = max(1, total_frames // 50)
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Only analyze every 5th frame to improve performance
            if frame_count % 5 == 0 or frame_count == 1:
                # Detect violence in the current frame using the enhanced analyzer
                violence_score, person_count, is_violent, analysis_data = violence_analyzer.analyze_frame(frame)
                
                # Track violent frames
                if is_violent:
                    violence_frames += 1
                    
                    # Start a new violent segment or continue the current one
                    if current_segment is None:
                        current_segment = {
                            "start_frame": frame_count,
                            "start_time": frame_count / fps,
                            "scores": [violence_score],
                            "person_count": person_count
                        }
                        
                        # Save keyframe for the start of violent segment
                        keyframe_path = f"{keyframes_dir}/violent_start_{frame_count}.jpg"
                        cv2.imwrite(keyframe_path, frame)
                        
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
                
                # Get detections for drawing
                detections = detect_objects(frame)
                annotated_frame = draw_bounding_boxes(annotated_frame, detections, is_violent, analysis_data)
                
                # Add status text
                if person_count > 0:
                    status_text = f"Violence: {violence_score:.2f} ({person_count} people)" if is_violent else f"No Violence ({person_count} people)"
                else:
                    status_text = "No People Detected"
                    
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
                keyframe_path = f"{keyframes_dir}/sample_{frame_count}.jpg"
                cv2.imwrite(keyframe_path, frame)
                
                keyframes.append({
                    "local_path": keyframe_path,
                    "s3_key": f"annotated_videos/keyframes_{timestamp}/sample_{frame_count}.jpg",
                    "type": "sample",
                    "frame": frame_count,
                    "time": frame_count / fps,
                    "score": violence_score if 'violence_score' in locals() else 0.0
                })
                
                sample_results.append({
                    "frame": int(frame_count),
                    "time": float(frame_count / fps),
                    "violence_detected": bool(is_violent),
                    "score": float(violence_score),  # Convert numpy float32 to Python float
                    "person_count": int(person_count)  # Convert numpy int to Python int
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
                    "frame": int(kf["frame"]),
                    "time": float(kf["time"]),
                    "score": float(kf["score"])  # Convert numpy float32 to Python float
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
            "keyframes": keyframe_urls,
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze_video/{analysis_id}")
async def get_video_analysis(analysis_id: str):
    """Get existing video analysis by ID"""
    try:
        analysis = await db["analyses"].find_one({"analysis_id": analysis_id})
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
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
        cursor = detections_collection.find(
            {"score": {"$gte": 0.3, "$lte": 0.5}},
            {"_id": 0, "timestamp": 1, "score": 1, "image_path": 1}
        ).limit(5)
        
        samples = []
        async for doc in cursor:
            samples.append({
                "id": doc["timestamp"],
                "image": f"/{doc['image_path']}",
                "confidence": doc["score"]
            })
            
        if not samples:
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
        return []

@app.post("/label_sample")
async def label_sample(data: dict):
    """Label a sample as violent or non-violent for model retraining"""
    try:
        sample_id = data.get("id")
        is_violent = data.get("is_violent")
        
        if sample_id is None or is_violent is None:
            raise HTTPException(status_code=400, detail="Missing id or is_violent in request")
            
        print(f"Sample {sample_id} labeled as {'violent' if is_violent else 'non-violent'}")
        
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
        test_input = torch.rand(1, 3, 224, 224)
        with torch.no_grad():
            output = model(test_input)
        
        return {
            "status": "ok", 
            "model_loaded": True,
            "output_shape": list(output.shape),
            "model_file": "retrained_model_v3.pth",
            "model_exists": os.path.exists("retrained_model_v3.pth"),
            "violence_analyzer": "Enhanced with motion analysis"
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e)
        }

app.mount("/annotated_videos", StaticFiles(directory="annotated_videos"), name="annotated_videos")
app.mount("/detections", StaticFiles(directory="detections"), name="detections")

@app.on_event("startup")
async def startup_event():
    try:
        info = await client.server_info()
        print("Successfully connected to MongoDB")
        print(f"Server will bind to: 0.0.0.0:{os.environ.get('PORT', 8000)}")
        
        test_tensor = torch.zeros((1, 3, 224, 224))
        with torch.no_grad():
            test_output = model(test_tensor)
        print(f"Violence detection model loaded successfully. Output shape: {test_output.shape}")
        print("Enhanced violence detection with motion analysis initialized")
        
        if os.path.exists("detections"):
            print(f"Detections directory exists: {len(os.listdir('detections'))} files")
        if os.path.exists("annotated_videos"):
            print(f"Annotated videos directory exists: {len(os.listdir('annotated_videos'))} files")
            
    except Exception as e:
        print(f"Startup error: {e}")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 8000))
    )
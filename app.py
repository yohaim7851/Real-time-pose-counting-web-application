from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import torch
import tempfile
import os
import yaml
from typing import Dict, Any
import pandas as pd
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from model import PoseRAC, Action_trigger
import logging
import json
import base64
import asyncio
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="PoseRAC Exercise Counter", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

class ExerciseCounter:
    def __init__(self, config_path: str = "RepCount_pose_config.yaml"):
        self.config_path = config_path
        self.model = None
        self.index2action = {}
        self.config = None
        self.pose_tracker = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Real-time processing state
        self.pose_buffer = deque(maxlen=30)  # Buffer for last 30 frames
        self.current_exercise = "unknown"
        self.rep_count = 0
        self.confidence = 0.0
        self.action_triggers = {}
        self.classify_prob = 0.5
        self.curr_pose = 'holder'
        self.init_pose = 'pose_holder'
        
        self.load_config_and_model()
    
    def load_config_and_model(self):
        """Load configuration and model weights"""
        try:
            # Load config
            with open(self.config_path, "r") as fd:
                self.config = yaml.load(fd, Loader=yaml.FullLoader)
            
            # Load action labels
            csv_label_path = self.config['dataset']['csv_label_path']
            label_pd = pd.read_csv(csv_label_path)
            
            for label_i in range(len(label_pd.index)):
                one_data = label_pd.iloc[label_i]
                action = one_data['action']
                label = one_data['label']
                self.index2action[label] = action
            
            num_classes = len(self.index2action)
            logger.info(f"Loaded {num_classes} action classes: {self.index2action}")
            
            # Initialize model
            self.model = PoseRAC(
                None, None, None, None,
                dim=self.config['PoseRAC']['dim'],
                heads=self.config['PoseRAC']['heads'],
                enc_layer=self.config['PoseRAC']['enc_layer'],
                learning_rate=self.config['PoseRAC']['learning_rate'],
                seed=self.config['PoseRAC']['seed'],
                num_classes=num_classes,
                alpha=self.config['PoseRAC']['alpha']
            )
            
            # Load model weights
            weight_loaded = False
            weight_paths = ['best_weights_PoseRAC.pth', self.config.get('save_ckpt_path', 'new_weights.pth')]
            
            for weight_path in weight_paths:
                if os.path.exists(weight_path):
                    try:
                        weights = torch.load(weight_path, map_location='cpu')
                        if 'state_dict' in weights:
                            self.model.load_state_dict(weights['state_dict'])
                        else:
                            self.model.load_state_dict(weights)
                        logger.info(f"Loaded weights from {weight_path}")
                        weight_loaded = True
                        break
                    except Exception as e:
                        logger.error(f"Failed to load {weight_path}: {e}")
            
            if not weight_loaded:
                raise FileNotFoundError("No valid model weights found")
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def normalize_landmarks(self, all_landmarks):
        """Normalize pose landmarks"""
        x_max = np.expand_dims(np.max(all_landmarks[:, :, 0], axis=1), 1)
        x_min = np.expand_dims(np.min(all_landmarks[:, :, 0], axis=1), 1)
        
        y_max = np.expand_dims(np.max(all_landmarks[:, :, 1], axis=1), 1)
        y_min = np.expand_dims(np.min(all_landmarks[:, :, 1], axis=1), 1)
        
        z_max = np.expand_dims(np.max(all_landmarks[:, :, 2], axis=1), 1)
        z_min = np.expand_dims(np.min(all_landmarks[:, :, 2], axis=1), 1)
        
        all_landmarks[:, :, 0] = (all_landmarks[:, :, 0] - x_min) / (x_max - x_min + 1e-8)
        all_landmarks[:, :, 1] = (all_landmarks[:, :, 1] - y_min) / (y_max - y_min + 1e-8)
        all_landmarks[:, :, 2] = (all_landmarks[:, :, 2] - z_min) / (z_max - z_min + 1e-8)
        
        all_landmarks = all_landmarks.reshape(len(all_landmarks), -1)
        return all_landmarks
    
    def extract_poses_from_video(self, video_path: str):
        """Extract pose landmarks from video"""
        logger.debug(f"Opening video file: {video_path}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video properties: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        poses = []
        valid_pose_count = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip some frames if video is too long (sample every few frames)
            if frame_count > 300:  # If more than 10 seconds at 30fps
                if frame_idx % max(1, frame_count // 300) != 0:
                    continue
            
            try:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process pose
                results = self.pose_tracker.process(frame_rgb)
                
                if results.pose_landmarks:
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    poses.append(landmarks)
                    valid_pose_count += 1
                else:
                    # Skip frames without pose instead of adding zeros
                    logger.debug(f"No pose detected in frame {frame_idx}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error processing frame {frame_idx}: {e}")
                continue
        
        cap.release()
        
        logger.info(f"Extracted {valid_pose_count} valid poses from {frame_idx} frames")
        
        if len(poses) == 0:
            raise ValueError("No valid poses detected in video. Please ensure a person is clearly visible.")
        
        if len(poses) < 10:
            logger.warning(f"Only {len(poses)} poses detected. Results may be unreliable.")
        
        return np.array(poses).reshape(-1, 33, 3)
    
    def count_repetitions(self, video_path: str) -> Dict[str, Any]:
        """Count exercise repetitions from video"""
        try:
            # Extract poses
            all_landmarks = self.extract_poses_from_video(video_path)
            all_landmarks = self.normalize_landmarks(all_landmarks)
            
            # Convert to tensor
            all_landmarks_tensor = torch.from_numpy(all_landmarks).float()
            
            with torch.no_grad():
                all_output = torch.sigmoid(self.model(all_landmarks_tensor))
            
            # First, determine the most likely exercise type based on average confidence
            logger.debug("Determining exercise type from video")
            average_confidences = torch.mean(all_output, dim=0)
            best_index = torch.argmax(average_confidences).item()
            best_action = self.index2action[best_index]
            best_confidence = average_confidences[best_index].item()
            
            logger.info(f"Detected exercise: {best_action} with confidence: {best_confidence:.3f}")
            
            # Log all exercise confidences for debugging
            for index in self.index2action:
                action_name = self.index2action[index]
                conf = average_confidences[index].item()
                logger.debug(f"Exercise {action_name}: {conf:.3f}")
            
            enter_threshold = self.config['Action_trigger']['enter_threshold']
            exit_threshold = self.config['Action_trigger']['exit_threshold']
            momentum = self.config['Action_trigger']['momentum']
            
            # Count repetitions for the detected exercise
            logger.debug(f"Counting repetitions for {best_action}")
            
            # Initialize action triggers for the best action
            rep_trigger1 = Action_trigger(
                action_name=best_action,
                enter_threshold=enter_threshold,
                exit_threshold=exit_threshold
            )
            rep_trigger2 = Action_trigger(
                action_name=best_action,
                enter_threshold=enter_threshold,
                exit_threshold=exit_threshold
            )
            
            classify_prob = 0.5
            pose_count = 0
            curr_pose = 'holder'
            init_pose = 'pose_holder'
            
            # Process each frame for repetition counting
            for frame_idx, output in enumerate(all_output):
                output_numpy = output[best_index].detach().cpu().numpy()
                classify_prob = output_numpy * (1. - momentum) + momentum * classify_prob
                
                # Count repetitions
                salient1_triggered = rep_trigger1(classify_prob)
                reverse_classify_prob = 1 - classify_prob
                salient2_triggered = rep_trigger2(reverse_classify_prob)
                
                if init_pose == 'pose_holder':
                    if salient1_triggered:
                        init_pose = 'salient1'
                        logger.debug(f"Frame {frame_idx}: Initial pose set to salient1")
                    elif salient2_triggered:
                        init_pose = 'salient2'
                        logger.debug(f"Frame {frame_idx}: Initial pose set to salient2")
                
                # Count transitions
                if init_pose == 'salient1':
                    if curr_pose == 'salient1' and salient2_triggered:
                        pose_count += 1
                        logger.debug(f"Frame {frame_idx}: Repetition counted (salient1->salient2). Total: {pose_count}")
                else:
                    if curr_pose == 'salient2' and salient1_triggered:
                        pose_count += 1
                        logger.debug(f"Frame {frame_idx}: Repetition counted (salient2->salient1). Total: {pose_count}")
                
                # Update current pose
                if salient1_triggered:
                    curr_pose = 'salient1'
                elif salient2_triggered:
                    curr_pose = 'salient2'
            
            logger.info(f"Final count for {best_action}: {pose_count} repetitions")
            best_count = pose_count
            
            return {
                'exercise_type': best_action,
                'repetition_count': best_count,
                'confidence': float(best_confidence),
                'total_frames': len(all_landmarks),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in count_repetitions: {e}")
            return {
                'exercise_type': 'unknown',
                'repetition_count': 0,
                'confidence': 0.0,
                'total_frames': 0,
                'status': 'error',
                'error_message': str(e)
            }
    
    def reset_real_time_state(self):
        """Reset state for real-time processing"""
        self.pose_buffer.clear()
        self.current_exercise = "unknown"
        self.rep_count = 0
        self.confidence = 0.0
        self.action_triggers = {}
        self.classify_prob = 0.5
        self.curr_pose = 'holder'
        self.init_pose = 'pose_holder'
        
        # Initialize action triggers for all exercises
        for index in self.index2action:
            action_type = self.index2action[index]
            self.action_triggers[f"{action_type}_1"] = Action_trigger(
                action_name=action_type,
                enter_threshold=self.config['Action_trigger']['enter_threshold'],
                exit_threshold=self.config['Action_trigger']['exit_threshold']
            )
            self.action_triggers[f"{action_type}_2"] = Action_trigger(
                action_name=action_type,
                enter_threshold=self.config['Action_trigger']['enter_threshold'],
                exit_threshold=self.config['Action_trigger']['exit_threshold']
            )
    
    def process_frame_real_time(self, frame):
        """Process a single frame for real-time inference"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose_tracker.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract landmarks
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Add to buffer
                self.pose_buffer.append(landmarks)
                
                # If we have enough frames, run inference
                if len(self.pose_buffer) >= 10:  # Minimum frames for reliable detection
                    self._update_real_time_inference()
                
                # Draw pose on frame
                mp_drawing.draw_landmarks(
                    frame_rgb, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS
                )
            
            # Convert back to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Add exercise info overlay
            self._add_info_overlay(frame_bgr)
            
            return frame_bgr, bool(results.pose_landmarks)
            
        except Exception as e:
            logger.error(f"Error in real-time frame processing: {e}")
            return frame, False
    
    def _update_real_time_inference(self):
        """Update exercise inference with current buffer"""
        try:
            if len(self.pose_buffer) < 10:
                return
            
            # Convert buffer to numpy array
            poses = np.array(list(self.pose_buffer)).reshape(-1, 33, 3)
            normalized_poses = self.normalize_landmarks(poses)
            
            # Run inference
            with torch.no_grad():
                poses_tensor = torch.from_numpy(normalized_poses).float()
                outputs = torch.sigmoid(self.model(poses_tensor))
                
                # Get latest prediction
                latest_output = outputs[-1]
                
                # Find best action
                best_confidence = 0
                best_action = "unknown"
                best_index = -1
                
                for index in self.index2action:
                    confidence = latest_output[index].item()
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_action = self.index2action[index]
                        best_index = index
                
                self.current_exercise = best_action
                self.confidence = best_confidence
                
                # Update repetition counting
                if best_index >= 0:
                    self._update_rep_count(best_index, latest_output)
                    
        except Exception as e:
            logger.error(f"Error in real-time inference: {e}")
    
    def _update_rep_count(self, action_index, output):
        """Update repetition count based on action triggers"""
        try:
            action_type = self.index2action[action_index]
            momentum = self.config['Action_trigger']['momentum']
            
            output_numpy = output[action_index].detach().cpu().numpy()
            self.classify_prob = output_numpy * (1. - momentum) + momentum * self.classify_prob
            
            # Get action triggers
            trigger1_key = f"{action_type}_1"
            trigger2_key = f"{action_type}_2"
            
            if trigger1_key not in self.action_triggers:
                return
            
            # Count repetitions
            salient1_triggered = self.action_triggers[trigger1_key](self.classify_prob)
            reverse_classify_prob = 1 - self.classify_prob
            salient2_triggered = self.action_triggers[trigger2_key](reverse_classify_prob)
            
            if self.init_pose == 'pose_holder':
                if salient1_triggered:
                    self.init_pose = 'salient1'
                elif salient2_triggered:
                    self.init_pose = 'salient2'
            
            if self.init_pose == 'salient1':
                if self.curr_pose == 'salient1' and salient2_triggered:
                    self.rep_count += 1
            else:
                if self.curr_pose == 'salient2' and salient1_triggered:
                    self.rep_count += 1
            
            if salient1_triggered:
                self.curr_pose = 'salient1'
            elif salient2_triggered:
                self.curr_pose = 'salient2'
                
        except Exception as e:
            logger.error(f"Error updating rep count: {e}")
    
    def _add_info_overlay(self, frame):
        """Add exercise information overlay to frame"""
        height, width = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Exercise info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Exercise: {self.current_exercise.replace('_', ' ').title()}", 
                   (20, 40), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Count: {self.rep_count}", 
                   (20, 70), font, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {self.confidence:.2%}", 
                   (20, 100), font, 0.7, (255, 0, 255), 2)

# Global exercise counter instance
exercise_counter = None

@app.on_event("startup")
async def startup_event():
    global exercise_counter
    try:
        exercise_counter = ExerciseCounter()
        logger.info("Exercise counter initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize exercise counter: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "PoseRAC Exercise Counter API", "status": "running"}

@app.get("/real_time")
async def real_time_interface():
    """Serve the real-time HTML interface"""
    return FileResponse("static/real_time.html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": exercise_counter is not None,
        "available_exercises": list(exercise_counter.index2action.values()) if exercise_counter else []
    }

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    """Analyze uploaded video and count exercise repetitions"""
    logger.info(f"Received video upload: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    if exercise_counter is None:
        logger.error("Exercise counter not initialized")
        raise HTTPException(status_code=500, detail="Exercise counter not initialized")
    
    # Validate file type
    logger.debug(f"File content type: {file.content_type}")
    if not file.content_type or not file.content_type.startswith('video/'):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be a video")
    
    tmp_file_path = None
    try:
        # Save uploaded file temporarily
        logger.debug("Creating temporary file...")
        file_extension = os.path.splitext(file.filename)[1] if file.filename else '.mp4'
        if not file_extension:
            file_extension = '.mp4'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            logger.debug(f"Read {len(content)} bytes from uploaded file")
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        logger.info(f"Temporary file saved: {tmp_file_path}")
        
        # Process video
        logger.debug("Starting video processing...")
        result = exercise_counter.count_repetitions(tmp_file_path)
        logger.info(f"Video processing completed: {result}")
        
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
            logger.debug("Temporary file cleaned up")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up temporary file if it exists
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
            logger.debug("Temporary file cleaned up after error")
        
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/exercises")
async def get_available_exercises():
    """Get list of available exercise types"""
    if exercise_counter is None:
        raise HTTPException(status_code=500, detail="Exercise counter not initialized")
    
    return {
        "exercises": list(exercise_counter.index2action.values()),
        "total_count": len(exercise_counter.index2action)
    }

@app.websocket("/ws/real_time")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time pose analysis"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    if exercise_counter is None:
        await websocket.close(code=1011, reason="Exercise counter not initialized")
        return
    
    exercise_counter.reset_real_time_state()
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "frame":
                # Decode base64 image
                img_data = base64.b64decode(message["data"])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Process frame
                    processed_frame, pose_detected = exercise_counter.process_frame_real_time(frame)
                    
                    # Encode processed frame
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send response
                    response = {
                        "type": "result",
                        "frame": img_base64,
                        "exercise": exercise_counter.current_exercise,
                        "count": exercise_counter.rep_count,
                        "confidence": exercise_counter.confidence,
                        "pose_detected": pose_detected
                    }
                    
                    await websocket.send_text(json.dumps(response))
            
            elif message["type"] == "reset":
                exercise_counter.reset_real_time_state()
                await websocket.send_text(json.dumps({
                    "type": "reset_confirmed",
                    "status": "success"
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))

@app.post("/reset_real_time")
async def reset_real_time():
    """Reset real-time exercise counting state"""
    if exercise_counter is None:
        raise HTTPException(status_code=500, detail="Exercise counter not initialized")
    
    exercise_counter.reset_real_time_state()
    return {"status": "success", "message": "Real-time state reset"}

@app.get("/real_time_status")
async def get_real_time_status():
    """Get current real-time exercise status"""
    if exercise_counter is None:
        raise HTTPException(status_code=500, detail="Exercise counter not initialized")
    
    return {
        "exercise": exercise_counter.current_exercise,
        "count": exercise_counter.rep_count,
        "confidence": exercise_counter.confidence,
        "buffer_size": len(exercise_counter.pose_buffer)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
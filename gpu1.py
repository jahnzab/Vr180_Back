# GPU-optimized VR180 converter for Google Colab T4
import torch
import cv2
import numpy as np
import os
import tempfile
import shutil
import uuid
import threading
import asyncio
import subprocess
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from urllib.parse import unquote
from moviepy.editor import VideoFileClip
import gc
import psutil
from fastapi.responses import FileResponse
from fastapi import HTTPException, Depends
from urllib.parse import unquote
import os
from sqlalchemy.orm import Session
# Initialize FastAPI app

import os
import tempfile
import asyncio
import threading
import subprocess
import requests
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import httpx
import io
import os
from pydantic import BaseModel
import tempfile
import subprocess
import json
import asyncio
import sys
import threading



from PIL import Image
import cv2
import io
from datetime import datetime
# Configuration
DATABASE_URL = "postgresql://vr_180_login_register_user:AIybaThG9Eeu583TbHSZj0mNSoxEXQgD@dpg-d2t75u7fte5s739uksig-a.oregon-postgres.render.com/vr_180_login_register"
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class Conversion(Base):
    __tablename__ = "conversions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    original_filename = Column(String)
    output_path = Column(String)  # Path to the converted file
    file_size = Column(Integer, nullable=True)  # Size in bytes
    created_at = Column(DateTime, default=datetime.utcnow)

# Then update your save_to_history function
def save_to_history(db: Session, user_id: int, original_filename: str, output_path: str, file_size: int = None):
    conversion = Conversion(
        user_id=user_id,
        original_filename=original_filename,
        output_path=output_path,
        file_size=file_size,
        # created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))  # Add this line
    )
    db.add(conversion)
    db.commit()
    db.refresh(conversion)
    return conversion

# And your get_conversion_by_id function
def get_conversion_by_id(db: Session, conversion_id: int, user_id: int):
    return db.query(Conversion).filter(
        Conversion.id == conversion_id, 
        Conversion.user_id == user_id
    ).first()
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class FeedbackCreate(BaseModel):
    content: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

app = FastAPI(title="VR 180 Video Converter API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user















import cv2
from fastapi.security import HTTPBearer
import torch
import torch.nn.functional as F
import gc
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline

security = HTTPBearer()

# GPU Configuration for Colab T4
def setup_gpu_environment():
    """Setup optimal GPU environment for T4"""
    if torch.cuda.is_available():
        # T4 has 16GB VRAM, optimize accordingly for Depth-Anything-V2-Large
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        device = torch.device("cuda")
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU not available, using CPU")
    return device

device = setup_gpu_environment()

# Load Depth-Anything-V2-Large model (best for VR 180 presence)
print("🔄 Loading Depth-Anything-V2-Large model...")
try:
    depth_estimator = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        device=0 if device.type == 'cuda' else -1,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    print("✅ Depth-Anything-V2-Large loaded successfully - Maximum VR presence quality")
except Exception as e:
    print(f"❌ Depth-Anything-V2-Large failed to load: {e}")
    # Fallback to smaller model
    try:
        depth_estimator = pipeline(
            task="depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Base-hf",
            device=0 if device.type == 'cuda' else -1
        )
        print("✅ Fallback to Depth-Anything-V2-Base")
    except Exception as e2:
        print(f"❌ All depth models failed: {e2}")
        depth_estimator = None

# Memory management utilities
def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_optimal_batch_size():
    """Optimal batch size for Depth-Anything-V2-Large"""
    if device.type == 'cuda':
        available_memory = torch.cuda.get_device_properties(0).total_memory
        # Depth-Anything-V2-Large is memory intensive
        if available_memory > 15e9:  # 15GB+ VRAM (T4)
            return 2  # Process 2 frames at a time for V2-Large
        else:
            return 1  # Conservative for smaller VRAM
    return 1

BATCH_SIZE = get_optimal_batch_size()
print(f"🎯 Using batch size: {BATCH_SIZE}")
def estimate_depth_batch(images):
    """
    FIXED: Process images one by one to avoid batch result parsing issues
    """
    if not images or depth_estimator is None:
        print("⚠️ No images or depth estimator unavailable")
        return [np.ones((480, 640), dtype=np.float32) * 0.5 for _ in images] if images else []
    
    print(f"🔍 Processing {len(images)} images individually")
    depths = []
    
    for i, img in enumerate(images):
        try:
            print(f"🔍 Processing image {i+1}/{len(images)}")
            
            # Validate image
            if img is None:
                print(f"⚠️ Image {i} is None")
                depths.append(np.ones((480, 640), dtype=np.float32) * 0.5)
                continue
                
            # Check dimensions
            if hasattr(img, 'shape'):
                h, w = img.shape[:2]
                if h <= 0 or w <= 0:
                    print(f"❌ CRITICAL: Invalid dimensions {w}x{h} for image {i}")
                    depths.append(np.ones((480, 640), dtype=np.float32) * 0.5)
                    continue
                print(f"   Image dimensions: {w}x{h}")
            
            # Convert to PIL RGB
            if isinstance(img, np.ndarray):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
            else:
                pil_image = img
            
            # CRITICAL: Process ONE image at a time
            result = depth_estimator(pil_image)
            
            # Handle result structure
            if isinstance(result, dict) and 'depth' in result:
                depth_array = np.array(result['depth'])
            elif hasattr(result, 'depth'):
                depth_array = np.array(result.depth)
            else:
                depth_array = np.array(result)
            
            print(f"   Depth array shape: {depth_array.shape}")
            
            # Validate depth array
            if depth_array.size == 0:
                raise ValueError("Empty depth array")
            
            # Normalize
            depth_min, depth_max = depth_array.min(), depth_array.max()
            if depth_max > depth_min:
                depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.ones_like(depth_array) * 0.5
                
            depths.append(depth_normalized.astype(np.float32))
            print(f"✅ Success for image {i}")
            
        except Exception as e:
            print(f"❌ Failed image {i}: {e}")
            # Fallback
            try:
                h, w = img.shape[:2] if hasattr(img, 'shape') else (480, 640)
                depths.append(np.ones((h, w), dtype=np.float32) * 0.5)
            except:
                depths.append(np.ones((480, 640), dtype=np.float32) * 0.5)
                
        # Clear GPU cache after each image
        clear_gpu_cache()
    
    print(f"✅ Completed: {len(depths)} depth maps")
    return depths
def temporal_smooth_gpu(prev_depths, curr_depths, alpha=0.9):  # Higher alpha for VR stability
    """GPU-accelerated temporal smoothing - OPTIMIZED VERSION"""
    if prev_depths is None or len(prev_depths) == 0 or curr_depths is None:
        return curr_depths if curr_depths is not None else []
    
    # Convert to tensors for batch processing
    try:
        prev_tensor = torch.from_numpy(np.array(prev_depths)).float().to(device)
        curr_tensor = torch.from_numpy(np.array(curr_depths)).float().to(device)
        
        # Ensure same length
        min_len = min(len(prev_tensor), len(curr_tensor))
        prev_tensor = prev_tensor[:min_len]
        curr_tensor = curr_tensor[:min_len]
        
        # Batch temporal smoothing on GPU
        smoothed_tensor = alpha * prev_tensor + (1 - alpha) * curr_tensor
        
        # Convert back to numpy
        smoothed = smoothed_tensor.cpu().numpy()
        
        # Handle remaining frames
        if len(curr_depths) > min_len:
            smoothed = list(smoothed) + list(curr_depths[min_len:])
        
        return smoothed
        
    except Exception as e:
        print(f"❌ GPU temporal smoothing failed: {e}")
        # Fallback: simple list-based smoothing
        min_len = min(len(prev_depths), len(curr_depths))
        smoothed = [alpha * prev_depths[i] + (1 - alpha) * curr_depths[i] for i in range(min_len)]
        if len(curr_depths) > min_len:
            smoothed.extend(curr_depths[min_len:])
        return smoothed
def estimate_single_depth(image):
    """
    Enhanced single frame depth estimation with comprehensive validation
    """
    if depth_estimator is None:
        print("⚠️ Depth estimator not available")
        h, w = image.shape[:2] if hasattr(image, 'shape') else (480, 640)
        return np.zeros((h, w), dtype=np.float32)
    
    try:
        # Validate input image
        if image is None:
            raise ValueError("Image is None")
        
        # Check dimensions for numpy arrays
        if hasattr(image, 'shape'):
            if len(image.shape) < 2:
                raise ValueError(f"Invalid image shape: {image.shape}")
            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid dimensions: {w}x{h}")
        
        # Convert BGR to RGB if needed
        if isinstance(image, np.ndarray):
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
        else:
            pil_image = image
        
        # Run depth estimation
        result = depth_estimator(pil_image)
        
        # Handle different result structures
        if isinstance(result, dict) and 'depth' in result:
            depth = np.array(result['depth'])
        elif hasattr(result, 'depth'):
            depth = np.array(result.depth)
        else:
            depth = np.array(result)
        
        # Validate depth array
        if depth.size == 0:
            raise ValueError("Empty depth array returned")
        
        # Normalize for VR processing
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.ones_like(depth) * 0.5
        
        return depth_normalized.astype(np.float32)
        
    except Exception as e:
        print(f"❌ Single depth estimation failed: {e}")
        # Return fallback depth map
        try:
            h, w = image.shape[:2] if hasattr(image, 'shape') else (480, 640)
            return np.ones((h, w), dtype=np.float32) * 0.5
        except:
            return np.ones((480, 640), dtype=np.float32) * 0.5
print("🏆 Depth-Anything-V2-Large VR 180 System Ready!")
print("🎯 Optimized for maximum presence and immersion in VR 180")
print("📝 Functions available:")
print("   - estimate_depth_batch(images): Process multiple frames")
print("   - estimate_single_depth(image): Process single frame")
print("   - temporal_smooth_gpu(prev, curr): Smooth between frames")
print("🚀 Ready for VR 180 depth processing!")
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt

# Add these endpoints to your existing /working endpoint
@app.get("/working")
async def working():
    gpu_status = "GPU Available" if torch.cuda.is_available() else "CPU Only"
    memory_info = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
    return {
        "message": "✅ System working - GPU optimized processing",
        "device": str(device),
        "gpu_status": gpu_status,
        "gpu_memory": memory_info,
        "batch_size": BATCH_SIZE,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "online"
    }

# Session keep-alive endpoint
@app.post("/auth/keep-alive")
async def keep_alive(current_user: User = Depends(get_current_user)):
    """Refresh session activity to prevent timeout"""
    return {
        "status": "active",
        "user_id": current_user.id,
        "username": current_user.username,
        "last_active": datetime.utcnow().isoformat(),
        "message": "Session kept alive successfully",
        "expires_in": "3 hours"
    }

# Session validation endpoint
@app.get("/auth/session-status")
async def session_status(current_user: User = Depends(get_current_user)):
    """Check if session is still valid"""
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "session_active": True,
        "timestamp": datetime.utcnow().isoformat()
    }

# Token refresh endpoint
@app.post("/auth/refresh-token")
async def refresh_token(refresh_data: dict, db: Session = Depends(get_db)):
    """Refresh access token using refresh token"""
    # Your token refresh implementation here
    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in": 10800  # 3 hours
    }
# -------------------------------
# GPU-Optimized Depth estimation
# -------------------------------
def inject_vr180_metadata_8k(
    input_video_path: str,
    output_video_path: str,
    center_scale_width: int = 8192,    # Full 8K width
    center_scale_height: int = 4096,   # VR180 standard height
    cropped_image_width: int = 8192,
    cropped_image_height: int = 4096,
    cropped_left: int = 0,
    cropped_top: int = 0
):

    """
    Inject VR180 metadata into a video using multiple approaches with built-in verification and debugging.
    Targets tags at the VIDEO STREAM level, which is most reliable for VR players.
    """
    import subprocess
    import json
    import os
    
    def debug_video_metadata(video_path: str):
        """Debug function to show all metadata in a video file."""
        print(f"🔍 Debugging metadata for: {video_path}")
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format:stream:format_tags:stream_tags",
                "-of", "json", video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            print("📋 All metadata found:")
            print(json.dumps(data, indent=2))
            return data
        except Exception as e:
            print(f"❌ Debug failed: {str(e)}")
            return None

    def verify_vr180_metadata(video_path: str) -> dict:
        """Verify if VR180 metadata is present and return detailed results."""
        try:
            # Method 1: Check with ffprobe
            probe_cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "stream_tags:format_tags",
                "-of", "json", video_path
            ]
            
            probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            metadata = json.loads(probe.stdout)
            
            metadata_str = str(metadata).lower()
            
            vr_indicators = [
                "spherical", "stereo_mode", "projection", 
                "equirectangular", "left_right", "spatial",
                "full_pano_width_pixels", "full_pano_height_pixels", 
                "cropped_area_image_width", "cropped_area_image_height"
            ]
            
            found_indicators = [indicator for indicator in vr_indicators if indicator in metadata_str]
            
            # Extract actual values (optional for detailed validation)
            extracted_values = {}
            try:
                format_tags = metadata.get('format', {}).get('tags', {})
                stream_tags = metadata.get('streams', [{}])[0].get('tags', {})
                
                # Combine and prioritize stream tags, as they are more important for players
                all_tags = {**format_tags, **stream_tags}
                
                extracted_values = {
                    "spherical": all_tags.get('spherical'),
                    "projection": all_tags.get('projection'),
                    "stereo_mode": all_tags.get('stereo_mode'),
                    "full_pano_width_pixels": all_tags.get('full_pano_width_pixels'),
                    "full_pano_height_pixels": all_tags.get('full_pano_height_pixels'),
                    "cropped_area_image_width": all_tags.get('cropped_area_image_width'),
                    "cropped_area_image_height": all_tags.get('cropped_area_image_height')
                }
            except Exception as e:
                print(f"⚠️ Failed to extract detailed metadata fields: {e}")

            if found_indicators:
                print(f"✅ VR180 metadata found: {', '.join(found_indicators)}")
                return {
                    "success": True,
                    "indicators": found_indicators,
                    "extracted_values": extracted_values,
                    "metadata": metadata
                }
            
            print(f"⚠️ No VR180 metadata detected in {video_path}")
            return {"success": False, "indicators": [], "metadata": metadata}

        except Exception as e:
            print(f"❌ Error verifying metadata: {str(e)}")
            return {"success": False, "error": str(e)}

    
    try:
        # Debug input file first
        print("🔍 DEBUGGING INPUT FILE:")
        debug_video_metadata(input_video_path)
        
        # Method 1: PRIMARY METHOD - Target VIDEO STREAM tags explicitly
        print("\n🔄 Attempting Method 1: Targeted Video Stream Metadata...")
        cmd1 = [
            "ffmpeg", "-y", "-i", input_video_path,
            "-c:v", "copy", "-c:a", "copy",        # Copy audio and video streams
            "-movflags", "use_metadata_tags",      # Important: use existing metadata tags
            "-map_metadata", "0",                  # Copy metadata from input to output
            # TARGET METADATA AT THE VIDEO STREAM LEVEL (s:v:0)
            "-metadata:s:v:0", f"spherical=true",
            "-metadata:s:v:0", f"stereo_mode=left_right",
            "-metadata:s:v:0", f"projection=equirectangular",
            "-metadata:s:v:0", f"full_pano_width_pixels={center_scale_width}",
            "-metadata:s:v:0", f"full_pano_height_pixels={center_scale_height}",
            "-metadata:s:v:0", f"cropped_area_image_width={cropped_image_width}",
            "-metadata:s:v:0", f"cropped_area_image_height={cropped_image_height}",
            "-metadata:s:v:0", f"cropped_area_left={cropped_left}",
            "-metadata:s:v:0", f"cropped_area_top={cropped_top}",
            output_video_path
        ]
        
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        
        if result1.returncode == 0:
            print("✅ Method 1 (Video Stream) command executed successfully.")
            print("🔍 DEBUGGING OUTPUT FILE AFTER METHOD 1:")
            verification = verify_vr180_metadata(output_video_path)
            if verification["success"]:
                print("🎉 SUCCESS: VR180 metadata verified!")
                return {"status": "success", "method": "targeted_stream", "output_file": output_video_path, "verification": verification}
            else:
                print("❌ Method 1 produced a file, but verification failed.")
        else:
            print(f"❌ Method 1 failed: {result1.stderr}")
        # Method 2: Using spatial-media approach (Google's standard)
        print("🔄 Attempting Method 2: Spatial media format...")
        cmd2 = [
            "ffmpeg", "-y", "-i", input_video_path,
            "-c:v", "copy", "-c:a", "copy",
            "-movflags", "faststart",
            "-metadata", "spherical-video=true",
            "-metadata", "stereo-mode=left-right",
            "-metadata", "source-count=2",
            output_video_path
        ]
        
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        
        if result2.returncode == 0:
            print("🔍 DEBUGGING OUTPUT FILE AFTER METHOD 2:")
            debug_video_metadata(output_video_path)
            verification = verify_vr180_metadata(output_video_path)
            if verification["success"]:
                return {"status": "success", "method": "spatial-media", "output_file": output_video_path, "verification": verification}
        else:
            print(f"❌ Method 2 failed: {result2.stderr}")
        
        print("\n⚠️ Method 2 failed, trying Method 3...")
        
        # Method 3: Explicit box writing for MP4
        print("🔄 Attempting Method 3: Direct MP4 box injection...")
        temp_output = output_video_path.replace('.mp4', '_temp.mp4')
        
        cmd3 = [
            "ffmpeg", "-y", "-i", input_video_path,
            "-c", "copy",
            "-map_metadata", "0",
            "-movflags", "+faststart",
            temp_output
        ]
        
        result3 = subprocess.run(cmd3, capture_output=True, text=True)
        
        if result3.returncode == 0:
            # Now add metadata using a different approach
            cmd3b = [
                "ffmpeg", "-y", "-i", temp_output,
                "-c", "copy",
                "-metadata:s:v:0", "handler_name=VideoHandler",
                "-metadata:s:v:0", "spherical=true",
                "-metadata:s:v:0", "stereo_mode=left_right",
                "-f", "mp4",
                output_video_path
            ]
            
            result3b = subprocess.run(cmd3b, capture_output=True, text=True)
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            if result3b.returncode == 0:
                print("🔍 DEBUGGING OUTPUT FILE AFTER METHOD 3:")
                debug_video_metadata(output_video_path)
                verification = verify_vr180_metadata(output_video_path)
                if verification["success"]:
                    return {"status": "success", "method": "direct-box", "output_file": output_video_path, "verification": verification}
            else:
                print(f"❌ Method 3b failed: {result3b.stderr}")
        else:
            print(f"❌ Method 3a failed: {result3.stderr}")
        
        print("\n⚠️ Method 3 failed, trying Method 4...")
        
        # Method 4: Using exiftool as fallback (if available)
        try:
            print("🔄 Attempting Method 4: ExifTool approach...")
            # First copy the file
            subprocess.run(["cp", input_video_path, output_video_path], check=True)
            
            # Then use exiftool to add metadata
            exiftool_cmd = [
                "exiftool", 
                "-overwrite_original",
                "-ProjectionType=equirectangular",
                "-StereoMode=left_right",
                "-SphericalVideo=true",
                output_video_path
            ]
            
            result4 = subprocess.run(exiftool_cmd, capture_output=True, text=True)
            
            if result4.returncode == 0:
                print("🔍 DEBUGGING OUTPUT FILE AFTER METHOD 4:")
                debug_video_metadata(output_video_path)
                verification = verify_vr180_metadata(output_video_path)
                if verification["success"]:
                    return {"status": "success", "method": "exiftool", "output_file": output_video_path, "verification": verification}
            else:
                print(f"❌ Method 4 failed: {result4.stderr}")
        except FileNotFoundError:
            print("📝 ExifTool not available, skipping Method 4")
        except Exception as e:
            print(f"❌ Method 4 exception: {str(e)}")
        
        # If all methods fail, still debug the final output
        print("\n🔍 FINAL DEBUG - ALL METHODS FAILED:")
        final_debug = debug_video_metadata(output_video_path)
        final_verification = verify_vr180_metadata(output_video_path)
        
        print("⚠️ All metadata injection methods failed")
        return {
            "status": "warning", 
            "output_file": output_video_path,
            "message": "File created but VR180 metadata may not be properly embedded",
            "final_metadata": final_debug,
            "verification": final_verification
        }
        
    except Exception as e:
        print(f"❌ Unexpected error in main function: {str(e)}")
        if os.path.exists(output_video_path):
            print("🔍 DEBUGGING OUTPUT FILE AFTER ERROR:")
            debug_video_metadata(output_video_path)
        raise RuntimeError(f"VR180 metadata injection failed: {str(e)}")
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np

# Initialize AI outpainting model (load this once at startup)
def init_outpainting_model():
    """Initialize AI outpainting model for peripheral extension"""
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("✅ AI Outpainting model loaded")
        return pipe
    except Exception as e:
        print(f"❌ Failed to load outpainting model: {e}")
        return None

outpainting_pipe = init_outpainting_model()

def ai_outpaint_frame(frame, expansion_percent=20):
    """
    Use AI to extend the frame's periphery for more natural VR180 edges
    """
    if outpainting_pipe is None:
        return frame  # Fallback to original frame
    
    # Convert to PIL if needed
    if isinstance(frame, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        pil_image = frame
    
    # Calculate expansion pixels
    width, height = pil_image.size
    expand_w = int(width * expansion_percent / 100)
    expand_h = int(height * expansion_percent / 100)
    
    # Create expanded canvas
    new_width = width + expand_w * 2
    new_height = height + expand_h * 2
    expanded_canvas = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    
    # Paste original image in center
    expanded_canvas.paste(pil_image, (expand_w, expand_h))
    
    # Create mask (black where we want to outpaint)
    mask = Image.new("L", (new_width, new_height), 255)  # White = keep
    # Create black rectangle where original image is (we want to preserve this)
    draw_mask = Image.new("L", (width, height), 0)  # Black = inpaint
    mask.paste(draw_mask, (expand_w, expand_h))
    
    # AI outpainting prompt (customize based on your content)
    prompt = "realistic environment extension, natural scenery, seamless blend, high quality"
    
    # Run outpainting
    result = outpainting_pipe(
        prompt=prompt,
        image=expanded_canvas,
        mask_image=mask,
        strength=0.9,
        guidance_scale=7.5,
        num_inference_steps=20
    ).images[0]
    
    # Crop back to original size or keep expanded based on your needs
    return result        
import cv2
import numpy as np

import numpy as np
import cv2

def make_stereo_pair_optimized(
    img, 
    depth, 
    eye_offset=8,          # ← KEEP (good default)
    center_scale=0.85,     # ← CHANGE from 0.9 to 0.85
    panini_alpha=0.6,      # ← CHANGE from 0.5 to 0.6
    max_blur_radius=12     # ← Fixed parameter name to match your pipeline
):
    """
    Creates a comfortable, immersive VR180 stereo pair from a 2D image and depth map.
    
    Args:
        img: Input image (2D or 3D array)
        depth: Depth map (same aspect ratio as img)
        eye_offset: Interocular distance (world units) - default: 8
        center_scale: Central scene scaling (0.8-0.95) - default: 0.85
        panini_alpha: Panini projection parameter (0.3-0.7) - default: 0.6
        max_blur_radius: Maximum blur radius for foveated effect - default: 12
    
    Returns:
        left, right: Stereo image pair
    """
    # Input validation to prevent OpenCV errors
    if img is None:
        raise ValueError("Input image is None")
    if depth is None:
        raise ValueError("Depth map is None")
    
    # Ensure img is numpy array
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Image must be numpy array, got {type(img)}")
    
    # Validate dimensions
    if len(img.shape) < 2:
        raise ValueError(f"Image must have at least 2 dimensions, got {len(img.shape)}")
    
    h, w = img.shape[:2]
    
    # Check for valid dimensions (this fixes the OpenCV resize error)
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")
    
    cx, cy = w // 2, h // 2
    epsilon = 1e-6
    focal_length = 500  # Fixed focal length
    
    # Handle depth map - ensure it's numpy array and correct type
    if not isinstance(depth, np.ndarray):
        depth = np.array(depth)
    
    # Normalize depth if it's not already in [0,1] range
    if depth.max() > 1.0:
        depth = depth.astype(np.float32) / 255.0
    
    # 1. Resize depth to match frame dimensions if needed
    if depth.shape[:2] != (h, w):
        try:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            print(f"⚠️ Depth resize failed: {e}")
            # Create fallback depth map
            depth = np.ones((h, w), dtype=np.float32) * 0.5
    
    # 2. Apply central scene scaling
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_scaled = ((x_coords - cx) * center_scale + cx).astype(np.int32)
    y_scaled = ((y_coords - cy) * center_scale + cy).astype(np.int32)
    x_scaled = np.clip(x_scaled, 0, w - 1)
    y_scaled = np.clip(y_scaled, 0, h - 1)
    
    # Apply scaling
    img_scaled = img[y_scaled, x_scaled]
    depth_scaled = depth[y_scaled, x_scaled]

    # 3. Calculate proper disparity from depth (physically based)
    depth_normalized = (depth_scaled - depth_scaled.min()) / (depth_scaled.max() - depth_scaled.min() + epsilon)
    depth_normalized = np.clip(depth_normalized, 0.1, 1.0)  # Avoid extreme values
    
    disparity = (eye_offset * focal_length) / depth_normalized
    max_disparity = w * 0.1  # Limit to 10% of screen width
    disparity = np.clip(disparity, 0, max_disparity).astype(np.int32)

    # 4. Create stereo images with depth-based shifting
    left = np.zeros_like(img_scaled)
    right = np.zeros_like(img_scaled)
    
    left_x = np.clip(x_coords - disparity // 2, 0, w - 1)
    right_x = np.clip(x_coords + disparity // 2, 0, w - 1)
    
    if len(img_scaled.shape) == 3:
        left[y_coords, left_x, :] = img_scaled[y_coords, x_coords, :]
        right[y_coords, right_x, :] = img_scaled[y_coords, x_coords, :]
    else:
        left[y_coords, left_x] = img_scaled[y_coords, x_coords]
        right[y_coords, right_x] = img_scaled[y_coords, x_coords]

    # 5. Apply Panini projection to both eyes (using panini_alpha as 'd' parameter)
    def apply_panini_projection(image, focal_length, d):
        """Apply correct Panini projection to reduce edge distortion."""
        if image is None or image.size == 0:
            return image
            
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return image
            
        cx, cy = w // 2, h // 2
        
        # Create coordinate grids
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x -= cx
        y -= cy
        
        # Panini projection equations
        ru = np.sqrt(x**2 + focal_length**2)
        sin_theta = x / ru
        tan_phi = y / (ru * d + focal_length * (1 - d))
        
        # Map to output
        map_x = (focal_length * sin_theta + cx)
        map_y = (focal_length * tan_phi + cy)
        
        try:
            return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        except cv2.error as e:
            print(f"⚠️ Panini projection failed: {e}")
            return image

    left = apply_panini_projection(left, focal_length, panini_alpha)
    right = apply_panini_projection(right, focal_length, panini_alpha)

    # 6. Apply fast foveated blur
    def foveated_blur_fast(image, center, max_blur_radius):
        """Efficient single-pass foveated blur."""
        if image is None or image.size == 0 or max_blur_radius <= 0:
            return image
            
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return image
        
        try:
            # Calculate distance map from center
            y_coords, x_coords = np.indices((h, w))
            dist = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)
            max_dist = np.sqrt(center[0]**2 + center[1]**2)
            
            if max_dist == 0:
                return image
                
            norm_dist = np.clip(dist / max_dist, 0, 1)
            
            # Use single large Gaussian blur
            max_kernel_size = max_blur_radius * 2 + 1
            if max_kernel_size % 2 == 0:  # Ensure odd kernel size
                max_kernel_size += 1
                
            blurred = cv2.GaussianBlur(image, (max_kernel_size, max_kernel_size), 0)
            
            # Create weight map for blending
            blend_weight = np.clip(norm_dist * 2, 0, 1)  # Linear transition
            
            # Blend based on weight map
            if len(image.shape) == 3:
                blend_weight = blend_weight[:, :, np.newaxis]
            
            result = (1 - blend_weight) * image + blend_weight * blurred
            return result.astype(image.dtype)
            
        except Exception as e:
            print(f"⚠️ Foveated blur failed: {e}")
            return image

    left = foveated_blur_fast(left, (cx, cy), max_blur_radius)
    right = foveated_blur_fast(right, (cx, cy), max_blur_radius)

    return left, right


def smart_upscale_to_8k(image, method="lanczos"):
    """
    Enhanced upscaling for 1K to 8K (4x scaling)
    """
    if image is None:
        raise ValueError("Image cannot be None")
    
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be numpy array, got {type(image)}")
    
    if len(image.shape) < 2:
        raise ValueError(f"Image must have at least 2 dimensions, got {len(image.shape)}")
    
    h, w = image.shape[:2]
    
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")
    
    # FIXED: 4x upscaling for 1K to 8K (instead of 2x)
    target_w, target_h = w * 8, h * 8
    
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target dimensions: {target_w}x{target_h}")
    
    # Select interpolation method
    if method == "lanczos":
        interpolation = cv2.INTER_LANCZOS4
    elif method == "cubic":
        interpolation = cv2.INTER_CUBIC
    elif method == "nearest":
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR
    
    try:
        upscaled = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        return upscaled
    except cv2.error as e:
        print(f"⚠️ Upscaling failed with {method}, falling back to linear: {e}")
        upscaled = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return upscaled

# Enhanced batch processing function
def process_stereo_batch_safe(frames, depth_maps, **kwargs):
    """
    Safely process a batch of frames and depth maps with error handling
    """
    if not frames or not depth_maps:
        return []
    
    if len(frames) != len(depth_maps):
        print(f"⚠️ Frame count ({len(frames)}) doesn't match depth count ({len(depth_maps)})")
        min_len = min(len(frames), len(depth_maps))
        frames = frames[:min_len]
        depth_maps = depth_maps[:min_len]
    
    stereo_pairs = []
    
    for i, (frame, depth) in enumerate(zip(frames, depth_maps)):
        try:
            # Validate inputs before processing
            if frame is None:
                print(f"⚠️ Frame {i} is None, skipping")
                continue
            if depth is None:
                print(f"⚠️ Depth {i} is None, creating fallback")
                h, w = frame.shape[:2]
                depth = np.ones((h, w), dtype=np.float32) * 0.5
            
            # Create stereo pair
            left, right = make_stereo_pair_optimized(frame, depth, **kwargs)
            
            # Combine into side-by-side format
            sbs = np.concatenate((left, right), axis=1)
            stereo_pairs.append(sbs)
            
        except Exception as e:
            print(f"⚠️ Failed to process stereo pair {i}: {e}")
            # Fallback: duplicate frame side by side
            try:
                if frame is not None:
                    sbs = np.concatenate((frame, frame), axis=1)
                    stereo_pairs.append(sbs)
            except Exception as e2:
                print(f"❌ Even fallback failed for frame {i}: {e2}")
                continue
    
    return stereo_pairs

print("🎯 Fixed VR 180 Stereo System Ready!")
print("✅ Enhanced error handling to prevent OpenCV resize errors")
print("✅ Input validation for all functions")
print("✅ Safe batch processing with fallbacks")
print("🚀 Ready for integration with your pipeline!")# Then define your conversion function AFTER the helper functions
def save_to_history(db: Session, user_id: int, original_filename: str, output_path: str, file_size: int = None):
    conversion = Conversion(
        user_id=user_id,
        original_filename=original_filename,
        output_path=output_path,
        file_size=file_size,
        created_at=datetime.utcnow()
    )
    db.add(conversion)
    db.commit()
    db.refresh(conversion)
    return conversion

    # Your conversion code here...
# upscaling to 8k
def convert_2d_to_vr180_gpu_optimized(
    input_path: str,
    output_path: str,
    center_scale: float = 0.9,
    focal_length: int = 500,
    panini_alpha: float = 0.5,
    eye_offset: int = 8,
    max_blur: int = 15,
    ai_outpaint: bool = True,
    upscale_to_8k: bool = True,
    upscale_method: str = "lanczos"
):
    """
    GPU-optimized 2D to VR180 conversion with enhanced error handling
    """
    # Add debug logging at start
    print("🔍 CONVERSION DEBUG MODE ENABLED")
    
    # Test video file first
    test_cap = cv2.VideoCapture(input_path)
    ret_test, frame_test = test_cap.read()
    if ret_test and frame_test is not None:
        print(f"✅ First frame OK: {frame_test.shape}")
    else:
        print("❌ CRITICAL: Cannot read first frame!")
        test_cap.release()
        raise ValueError("Cannot read video frames")
    test_cap.release()
    
    outpainting_pipe = None
    if ai_outpaint:
        try:
            outpainting_pipe = init_outpainting_model()
        except Exception as e:
            print(f"⚠️ AI outpainting initialization failed: {e}")
            ai_outpaint = False
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Get video properties with validation
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # FIXED: Proper indentation and validation
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions: {width}x{height}")
        if total_frames <= 0:
            raise ValueError(f"Invalid frame count: {total_frames}")
        if fps <= 0:
            fps = 25.0  # Default fallback

        print(f"📹 Processing video: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")
        
        if ai_outpaint and outpainting_pipe:
            print("🎨 AI Outpainting: ENABLED")
        else:
            print("🎨 AI Outpainting: DISABLED")
        
        if upscale_to_8k:
            print(f"🔍 8K Upscaling: ENABLED ({upscale_method})")
        else:
            print("🔍 8K Upscaling: DISABLED")

        frame_paths = []
        prev_depths = None
        frame_batch = []
        processed_frames = 0
        
        # FIXED: Use local batch size instead of modifying global
        current_batch_size = max(1, BATCH_SIZE // 2) if upscale_to_8k else BATCH_SIZE
        print(f"📦 Using batch size: {current_batch_size}")

        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process final batch
                if frame_batch:
                    print(f"🔄 Processing final batch: {len(frame_batch)} frames")
                    
                    try:
                        depths = estimate_depth_batch(frame_batch)
                        
                        # Validate depths count
                        if len(depths) != len(frame_batch):
                            print(f"⚠️ Adjusting depths: {len(depths)} -> {len(frame_batch)}")
                            while len(depths) < len(frame_batch):
                                depths.append(np.ones((height, width), dtype=np.float32) * 0.5)
                            depths = depths[:len(frame_batch)]
                        
                        if prev_depths is not None:
                            depths = temporal_smooth_gpu(prev_depths, depths)
                        
                        # Process each frame with validation
                        for i, (frame_data, depth) in enumerate(zip(frame_batch, depths)):
                            try:
                                print(f"🔄 Processing frame {processed_frames + i + 1}/{total_frames}")
                                
                                # CRITICAL: Validate frame before processing
                                if frame_data is None:
                                    print(f"⚠️ Skipping None frame {processed_frames + i + 1}")
                                    continue
                                
                                if not isinstance(frame_data, np.ndarray):
                                    frame_data = np.array(frame_data)
                                
                                frame_h, frame_w = frame_data.shape[:2]
                                if frame_h <= 0 or frame_w <= 0:
                                    print(f"⚠️ Skipping invalid frame: {frame_w}x{frame_h}")
                                    continue
                                
                                # Apply 8K upscaling with validation
                                if upscale_to_8k:
                                    try:
                                        frame_data = smart_upscale_to_8k(frame_data, upscale_method)
                                        depth = smart_upscale_to_8k(depth, "nearest")
                                    except Exception as e:
                                        print(f"⚠️ Upscaling failed: {e}")
                                        continue
                                
                                # Create stereo pair with validation
                                try:
                                    left, right = make_stereo_pair_optimized(
                                        frame_data, 
                                        depth,
                                        eye_offset=eye_offset,
                                        center_scale=center_scale,
                                        panini_alpha=panini_alpha,
                                        max_blur_radius=max_blur
                                    )
                                    
                                    if left is None or right is None:
                                        print(f"⚠️ Stereo creation returned None")
                                        continue
                                    
                                    sbs = np.concatenate((left, right), axis=1)
                                    
                                except Exception as e:
                                    print(f"⚠️ Stereo creation failed: {e}")
                                    continue
                                
                                # Save frame
                                frame_path = os.path.join(temp_dir, f"{len(frame_paths):05d}.png")
                                if cv2.imwrite(frame_path, sbs, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                                    frame_paths.append(frame_path)
                                    processed_frames += 1
                                    
                            except Exception as e:
                                print(f"❌ Frame processing failed: {e}")
                                continue
                        
                        clear_gpu_cache()
                        
                    except Exception as e:
                        print(f"❌ Final batch failed: {e}")
                        break
                
                break

            # CRITICAL: Validate incoming frame
            if frame is None:
                print(f"⚠️ Received None frame at position {len(frame_batch)}")
                continue
            
            if not isinstance(frame, np.ndarray):
                try:
                    frame = np.array(frame)
                except Exception as e:
                    print(f"⚠️ Frame conversion failed: {e}")
                    continue
            
            if len(frame.shape) < 2:
                print(f"⚠️ Invalid frame shape: {frame.shape}")
                continue
            
            frame_h, frame_w = frame.shape[:2]
            if frame_h <= 0 or frame_w <= 0:
                print(f"⚠️ Invalid frame dimensions: {frame_w}x{frame_h}")
                continue

            # AI Outpainting (with error handling)
            if ai_outpaint and outpainting_pipe is not None:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    outpainted_frame = ai_outpaint_frame(pil_frame, expansion_percent=15)
                    frame = cv2.cvtColor(np.array(outpainted_frame), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"⚠️ AI outpainting failed: {e}")

            frame_batch.append(frame)
            
            # Process batch when full
            if len(frame_batch) >= current_batch_size:
                print(f"🔄 Processing batch: {len(frame_batch)} frames")
                
                try:
                    depths = estimate_depth_batch(frame_batch)
                    
                    # Validate and adjust depths
                    if len(depths) != len(frame_batch):
                        print(f"⚠️ Adjusting depths: {len(depths)} -> {len(frame_batch)}")
                        while len(depths) < len(frame_batch):
                            depths.append(np.ones((height, width), dtype=np.float32) * 0.5)
                        depths = depths[:len(frame_batch)]
                    
                    if prev_depths is not None:
                        depths = temporal_smooth_gpu(prev_depths, depths)
                    
                    prev_depths = depths[-current_batch_size:]
                    
                    # Process batch with individual frame validation
                    for i, (frame_data, depth) in enumerate(zip(frame_batch, depths)):
                        try:
                            # Validate frame
                            if frame_data is None:
                                continue
                            
                            frame_h, frame_w = frame_data.shape[:2]
                            if frame_h <= 0 or frame_w <= 0:
                                continue
                            
                            # Apply 8K upscaling
                            if upscale_to_8k:
                                try:
                                    frame_data = smart_upscale_to_8k(frame_data, upscale_method)
                                    depth = smart_upscale_to_8k(depth, "nearest")
                                except Exception as e:
                                    print(f"⚠️ Upscaling failed: {e}")
                                    continue
                            
                            # Create stereo pair
                            left, right = make_stereo_pair_optimized(
                                frame_data, 
                                depth,
                                eye_offset=eye_offset,
                                center_scale=center_scale,
                                panini_alpha=panini_alpha,
                                max_blur_radius=max_blur
                            )
                            
                            sbs = np.concatenate((left, right), axis=1)
                            
                            frame_path = os.path.join(temp_dir, f"{len(frame_paths):05d}.png")
                            if cv2.imwrite(frame_path, sbs, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                                frame_paths.append(frame_path)
                                processed_frames += 1
                                
                        except Exception as e:
                            print(f"❌ Frame {processed_frames + i + 1} failed: {e}")
                            continue
                    
                    frame_batch = []
                    
                    if processed_frames % (current_batch_size * 3) == 0:
                        progress = (processed_frames / total_frames) * 100
                        print(f"📊 Progress: {progress:.1f}% ({processed_frames}/{total_frames})")
                        clear_gpu_cache()
                        
                except Exception as e:
                    print(f"❌ Batch processing failed: {e}")
                    frame_batch = []
                    continue

        cap.release()

        if not frame_paths:
            raise RuntimeError("No frames were successfully processed")

        print(f"✅ Processed {len(frame_paths)} frames, creating video...")
        
        # === FIXED VIDEO CREATION SECTION ===
        try:
            if upscale_to_8k:
                # For 8K, use software encoder (x264) as NVENC doesn't support >4096 width
                print("🎬 Using software encoder for 8K (NVENC limitation)")
                cmd = [
                    "ffmpeg", "-y",
                    "-r", str(fps),
                    "-i", os.path.join(temp_dir, "%05d.png"),
                    "-c:v", "libx264",           # Software encoder
                    "-preset", "medium",         # Balanced speed/quality
                    "-crf", "18",               # High quality
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",   # Web optimization
                    "-threads", "0",            # Use all CPU cores
                    output_path
                ]
                
                # Try with lower quality if encoding fails
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                
                if result.returncode != 0:
                    print("⚠️ High quality encoding failed, trying faster preset")
                    cmd[cmd.index("-preset") + 1] = "fast"
                    cmd[cmd.index("-crf") + 1] = "23"  # Lower quality
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                    
            else:
                # For 4K and below, try NVENC first, fallback to x264
                print("🎬 Trying hardware encoder (NVENC)")
                cmd = [
                    "ffmpeg", "-y",
                    "-r", str(fps),
                    "-i", os.path.join(temp_dir, "%05d.png"),
                    "-c:v", "h264_nvenc",
                    "-preset", "p4",             # NVENC preset
                    "-cq", "23",                # Constant quality
                    "-b:v", "25M",              # Target bitrate
                    "-maxrate", "35M",
                    "-bufsize", "50M",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-threads", "0",
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
                
                if result.returncode != 0:
                    print("⚠️ NVENC failed, falling back to software encoder")
                    cmd = [
                        "ffmpeg", "-y",
                        "-r", str(fps),
                        "-i", os.path.join(temp_dir, "%05d.png"),
                        "-c:v", "libx264",
                        "-preset", "medium",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        "-threads", "0",
                        output_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            
            if result.returncode != 0:
                # Final fallback with very safe settings
                print("⚠️ Standard encoding failed, trying safe fallback")
                cmd = [
                    "ffmpeg", "-y",
                    "-r", str(fps),
                    "-i", os.path.join(temp_dir, "%05d.png"),
                    "-c:v", "libx264",
                    "-preset", "ultrafast",      # Fastest encoding
                    "-crf", "28",               # Lower quality but reliable
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
                    output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
                
            if result.returncode != 0:
                raise RuntimeError(f"All encoding attempts failed. Last error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise RuntimeError("Video encoding timeout - try reducing quality or video length")

        if not os.path.exists(output_path):
            raise RuntimeError("Failed to create output video file")

        print("✅ VR180 conversion completed successfully")
        if upscale_to_8k:
            print("🎯 8K upscaling successfully applied")
            print("ℹ️  Note: 8K encoding uses software encoder (CPU) due to hardware limitations")
            
    finally:
        if 'cap' in locals():
            cap.release()
        shutil.rmtree(temp_dir, ignore_errors=True)
        clear_gpu_cache()
    # Create output video - adjust settings for 8K
   
# Base directories
BASE_DIR = "/content/Vr180_Back"  # Colab base directory
UPLOAD_DIR = os.path.join(BASE_DIR, "tmp_uploads")
FINAL_DIR = os.path.join(BASE_DIR, "videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

@app.get("/videos/{filename}")
async def get_video(
    filename: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Decode URL-encoded filename
    filename = unquote(filename)
    
    # Find the conversion record
    conversion = db.query(Conversion).filter(
        Conversion.output_path.like(f"%{filename}"),
        Conversion.user_id == current_user.id
    ).first()
    
    if not conversion:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = conversion.output_path
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    def iter_file(path: str):
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(
        iter_file(file_path), 
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="vr180_{conversion.original_filename}"'}
    )
def get_conversion_by_id(db: Session, conversion_id: int, user_id: int):
    return db.query(Conversion).filter(
        Conversion.id == conversion_id, 
        Conversion.user_id == user_id
    ).first()

@app.get("/download/{conversion_id}")
async def download_conversion(
    conversion_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Use the function to get the conversion
        conversion = get_conversion_by_id(db, conversion_id, current_user.id)
        
        if not conversion:
            raise HTTPException(status_code=404, detail="Conversion not found")
        
        if not conversion.output_path or not os.path.exists(conversion.output_path):
            raise HTTPException(status_code=404, detail="Converted file not found")
        
        # Return the file
        return FileResponse(
            conversion.output_path,
            media_type="video/mp4",
            filename=f"vr180_{conversion.original_filename}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Simplified auth for Colab

def save_to_history(db: Session, user_id: int, original_filename: str, output_path: str, file_size: int = None):
    conversion = Conversion(
        user_id=user_id,
        original_filename=original_filename,
        output_path=output_path,
        file_size=file_size,
        created_at=datetime.utcnow()
    )
    db.add(conversion)
    db.commit()
    db.refresh(conversion)
    return conversion

from fastapi import Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
  # Your database session   # Your auth dependency
@app.post("/convert/")
async def convert_video(
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Check file size (Colab has limited storage)
    content = await video.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > 500:  # 500MB limit for Colab
        raise HTTPException(status_code=400, detail="File too large. Maximum 500MB for Colab.")

    unique_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"input_{unique_id}.mp4")
    output_path = os.path.join(UPLOAD_DIR, f"output_{unique_id}_vr180.mp4")
    final_output_path = os.path.join(FINAL_DIR, f"vr180_{unique_id}.mp4")
    
    # Define temp_with_audio outside try block to avoid scope issues
    temp_with_audio = output_path.replace(".mp4", "_with_audio.mp4")

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(content)

    try:
        print(f"🚀 Starting conversion: {video.filename} ({file_size_mb:.1f}MB)")
        print(f"👤 User: {current_user.username} (ID: {current_user.id})")
        
        # Clear GPU memory before starting
        clear_gpu_cache()
        
        # Step 1: Convert to VR180 using GPU optimization - FIXED
        await asyncio.to_thread(
            convert_2d_to_vr180_gpu_optimized,
            input_path,
            output_path,
            center_scale=0.85,
            focal_length=550,
            panini_alpha=0.6,
            eye_offset=8,
            max_blur=12,
            ai_outpaint=True,
            upscale_to_8k=True,
            upscale_method="lanczos" 
            
        )

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="VR180 conversion failed.")

        # Step 2: Add audio with metadata preservation
        await add_audio_preserve_metadata(input_path, output_path, temp_with_audio)
        
        # Step 3: Inject VR180 metadata - FIXED
        # In your convert_video endpoint, update this section:
        await asyncio.to_thread(
    inject_vr180_metadata,
    temp_with_audio,       # input_video_path
    final_output_path,     # output_video_path
    7680,                  # center_scale_width (4x upscaled from 1920)
    4320,                  # center_scale_height (4x upscaled from 1080)  
    7680,                  # cropped_image_width (your actual output width)
    4320,                  # cropped_image_height (your actual output height)
    0,                     # cropped_left
    0                      # cropped_top (0 for side-by-side format)
)

        if not os.path.exists(final_output_path):
            raise HTTPException(status_code=500, detail="Final conversion failed.")

        final_file_size = os.path.getsize(final_output_path)
        
        # Step 4: Save to conversion history with error handling
        try:
            conversion = save_to_history(
                db=db,
                user_id=current_user.id,
                original_filename=video.filename,
                output_path=final_output_path,
                file_size=final_file_size
            )
            print(f"✅ Conversion saved to history with ID: {conversion.id}")
        except Exception as db_error:
            print(f"⚠️ Database save error: {db_error}")
            conversion = None

        # Step 5: Return file for download
        def iter_file(path: str):
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk

        final_size = os.path.getsize(final_output_path) / (1024 * 1024)
        print(f"✅ Conversion completed: {final_size:.1f}MB")

        headers = {
            "Content-Disposition": f'attachment; filename="vr180_{video.filename}"',
            "X-File-Size-MB": f"{final_size:.1f}"
        }
        
        if conversion:
            headers["X-Conversion-ID"] = str(conversion.id)

        return StreamingResponse(
            iter_file(final_output_path),
            media_type="video/mp4",
            headers=headers
        )

    except Exception as e:
        print(f"❌ Conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temporary files
        def cleanup():
           try:
              cleanup_paths = [input_path]   # only remove input file
              if 'temp_with_audio' in locals() and temp_with_audio:
                 cleanup_paths.append(temp_with_audio)  # remove temp audio if exists

              for path in cleanup_paths:
                if os.path.exists(path):
                  os.unlink(path)
                  print(f"🧹 Cleaned up: {path}")

              clear_gpu_cache()  # <-- keep this inside try
           except Exception as e:
               print(f"Cleanup error: {e}")

        threading.Thread(target=cleanup, daemon=True).start()
# Enhanced user conversions endpoint with better debugging
@app.get("/user/conversions/")
async def get_user_conversions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        print(f"📋 Fetching conversions for user: {current_user.username} (ID: {current_user.id})")
        
        # Get conversions from database for the current user
        conversions = db.query(Conversion).filter(
            Conversion.user_id == current_user.id
        ).order_by(Conversion.created_at.desc()).all()
        
        print(f"📊 Found {len(conversions)} conversions")
        
        # Convert to list of dictionaries with additional info
        result = []
        for conv in conversions:
            file_exists = os.path.exists(conv.output_path) if conv.output_path else False
            file_size_mb = conv.file_size / (1024 * 1024) if conv.file_size else None
            
            result.append({
                "id": conv.id,
                "original_filename": conv.original_filename,
                "created_at": conv.created_at,
                "file_size": conv.file_size,
                "file_size_mb": round(file_size_mb, 2) if file_size_mb else None,
                "file_exists": file_exists,
                "output_path": conv.output_path  # Include path for debugging
            })
        
        return {
            "conversions": result,
            "total": len(result),
            "user_id": current_user.id,
            "username": current_user.username
        }
        
    except Exception as e:
        print(f"❌ Error fetching conversions: {e}")
        return {
            "conversions": [],
            "total": 0,
            "error": str(e),
            "user_id": current_user.id if current_user else None
        }

# Debug endpoint to check database connection
@app.get("/debug/db-test")
async def test_database(db: Session = Depends(get_db)):
    try:
        # Test basic database query
        total_conversions = db.query(Conversion).count()
        total_users = db.query(User).count()
        
        return {
            "database_connected": True,
            "total_conversions": total_conversions,
            "total_users": total_users,
            "message": "Database is working"
        }
    except Exception as e:
        return {
            "database_connected": False,
            "error": str(e),
            "message": "Database connection failed"
        }

# Test endpoint to manually create a conversion record
@app.post("/debug/test-conversion")
async def test_create_conversion(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        test_conversion = save_to_history(
            db=db,
            user_id=current_user.id,
            original_filename="test_video.mp4",
            output_path="/fake/path/test_output.mp4",
            file_size=1024000
        )
        
        return {
            "success": True,
            "conversion_id": test_conversion.id,
            "message": "Test conversion created successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to create test conversion"
        }


async def add_audio_preserve_metadata(original_path, vr180_video_path, output_path):
    """Add audio to VR180 video while preserving metadata - Colab optimized"""
    cmd = [
        'ffmpeg', '-y',
        '-i', vr180_video_path,  # VR180 video with metadata
        '-i', original_path,     # Original video with audio
        '-map', '0:v',          # Use video from VR180
        '-map', '1:a',          # Use audio from original
        '-c:v', 'copy',         # Copy video (preserves metadata!)
        '-c:a', 'aac',          # Encode audio to AAC
        '-b:a', '128k',         # Reasonable audio bitrate for Colab
        '-movflags', 'use_metadata_tags+faststart',
        '-map_metadata', '0',   # Copy metadata from VR180 video
        '-threads', '0',        # Use all CPU cores
        output_path
    ]
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            print(f"FFmpeg error: {error_msg}")
            raise Exception(f"Audio addition failed: {error_msg}")
            
    except asyncio.TimeoutError:
        raise Exception("Audio processing timeout - video too long")
    except Exception as e:
        print(f"Audio addition error: {e}")
        # Fallback: copy VR180 video without audio
        shutil.copy2(vr180_video_path, output_path)
@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

from pydantic import BaseModel

class UserRegister(BaseModel):
    username: str
    email: str
    password: str
@app.post("/auth/register", response_model=Token)
async def register(user: UserRegister, db: Session = Depends(get_db)):
    # Check if username or email already exists
    existing_user = db.query(User).filter(
        (User.username == user.username) | (User.email == user.email)
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )

    # Create new user
    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        created_at=datetime.utcnow()
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Automatically generate JWT token after registration
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}

# Temporary debug endpoint
@app.get("/debug/conversions")
async def debug_conversions(db: Session = Depends(get_db)):
    all_conv = db.query(Conversion).all()
    return {"total": len(all_conv), "conversions": all_conv}
@app.get("/user/conversions/")
async def get_user_conversions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        print(f"🔍 [API Call] /user/conversions/ for user ID: {current_user.id}")
        
        # Get conversions from database for the current user
        conversions = db.query(Conversion).filter(
            Conversion.user_id == current_user.id
        ).order_by(Conversion.created_at.desc()).all()
        
        print(f"✅ [Database] Found {len(conversions)} conversions for user {current_user.id}")
        
        # Debug: Print each conversion found
        for i, conv in enumerate(conversions):
            print(f"   {i+1}. ID: {conv.id}, File: {conv.original_filename}, Size: {conv.file_size} bytes")
        
        # Convert to list of dictionaries for JSON response
        result = [
            {
                "id": conv.id,
                "user_id": conv.user_id,
                "original_filename": conv.original_filename,
                "converted_filename": os.path.basename(conv.output_path) if conv.output_path else "Unknown",
                "output_path": conv.output_path,
                "file_size_bytes": conv.file_size,
                "file_size_mb": f"{(conv.file_size or 0) / (1024 * 1024):.1f}MB" if conv.file_size else "N/A",
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "status": "completed",
                "download_url": f"/api/download/{conv.id}"
            }
            for conv in conversions
        ]
        
        print(f"📦 [Response] Returning {len(result)} items")
        return result
        
    except Exception as e:
        print(f"❌ [Error] Failed to fetch conversions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversion history: {str(e)}")
# @app.get("/user/conversions/")
# async def get_user_conversions(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     conversions = db.query(Conversion).filter(
#         Conversion.user_id == current_user.id
#     ).order_by(Conversion.created_at.desc()).all()
    
#     return [
#         {
#             "id": conv.id,
#             "original_filename": conv.original_filename,
#             "converted_filename": conv.converted_filename,
#             "created_at": conv.created_at
#         }
#         for conv in conversions
#     ]
from typing import List
@app.get("/admin/feedback/")
async def get_all_feedback(db: Session = Depends(get_db)):
    """Admin endpoint to view all feedback (add admin authentication in production)"""
    feedback_list = db.query(Feedback).join(User).order_by(Feedback.created_at.desc()).all()
    
    return [
        {
            "id": fb.id,
            "username": fb.user.username if hasattr(fb, 'user') else "Unknown",
            "content": fb.content,
            "created_at": fb.created_at
        }
        for fb in feedback_list
    ]
# In your FastAPI app, make sure you have:
@app.get("/feedback")
async def get_feedback(current_user: User = Depends(get_current_user)):
    return {"message": "Feedback endpoint works"}
class FeedbackOut(BaseModel):
    id: int
    user_id: int
    content: str
    created_at: datetime

    class Config:
        orm_mode = True  # important to convert SQLAlchemy objects to dict



@app.get("/feedback/", response_model=List[FeedbackOut])
def get_all_feedback(db: Session = Depends(get_db)):
    feedback_list = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
    return feedback_list
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


# Colab memory monitoring
@app.get("/system/status")
async def system_status():
    """Get system status for Colab monitoring"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB",
            "memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.1f}GB"
        }
    
    return {
        "gpu": gpu_info,
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent,
        "batch_size": BATCH_SIZE
    }

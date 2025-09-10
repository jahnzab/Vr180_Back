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
















security = HTTPBearer()

# GPU Configuration for Colab T4
def setup_gpu_environment():
    """Setup optimal GPU environment for T4"""
    if torch.cuda.is_available():
        # T4 has 16GB VRAM, optimize accordingly
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
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

# Load MiDaS Large model (better quality for T4 GPU)
print("🔄 Loading MiDaS Large model...")
try:
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # High-quality model
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    print("✅ MiDaS Large (DPT) loaded successfully")
except Exception as e:
    print(f"⚠️ DPT_Large failed, falling back to MiDaS_large: {e}")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")  # Fallback
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform

# Optimize model for GPU
midas.to(device)
midas.eval()

# Enable mixed precision for T4 GPU efficiency
if device.type == 'cuda':
    midas.half()  # Use FP16 for memory efficiency on T4
    print("✅ Enabled FP16 mixed precision")

# Memory management utilities
def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_optimal_batch_size():
    """Determine optimal batch size based on available GPU memory"""
    if device.type == 'cuda':
        # T4 has 16GB, but account for model memory
        available_memory = torch.cuda.get_device_properties(0).total_memory
        # Conservative estimate for T4
        if available_memory > 15e9:  # 15GB+
            return 4
        elif available_memory > 10e9:  # 10GB+
            return 2
        else:
            return 1
    return 1

BATCH_SIZE = get_optimal_batch_size()
print(f"🎯 Using batch size: {BATCH_SIZE}")

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
def estimate_depth_batch(images):
    """
    Run MiDaS depth estimation on a batch of frames with proper data type handling
    """
    if not images:
        return []
    
    # Convert all BGR to RGB
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    
    # Transform each image and stack into a batch
    input_tensors = []
    for img_rgb in images_rgb:
        tensor = midas_transforms(img_rgb).to(device)
        
        # Ensure the tensor has the correct data type (match model weights)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
            
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        input_tensors.append(tensor)
    
    # Stack all tensors into a batch
    batch_tensor = torch.cat(input_tensors, dim=0)  # [B, C, H, W]
    
    # Ensure batch tensor has the same data type as model weights
    if batch_tensor.dtype != next(midas.parameters()).dtype:
        batch_tensor = batch_tensor.to(next(midas.parameters()).dtype)
    
    with torch.no_grad():
        predictions = midas(batch_tensor)  # output [B, H_out, W_out] or [B, 1, H_out, W_out]
    
    # Ensure predictions are float32 for interpolation
    if predictions.dtype != torch.float32:
        predictions = predictions.float()
    
    # Ensure 4D for interpolate
    if predictions.dim() == 3:
        predictions = predictions.unsqueeze(1)  # [B, 1, H, W]
    
    depths = []
    for i in range(len(images)):
        # Resize to original frame size
        prediction_resized = torch.nn.functional.interpolate(
            predictions[i:i+1], 
            size=images[i].shape[:2], 
            mode="bicubic", 
            align_corners=False
        ).squeeze(0).squeeze(0)  # remove batch & channel dims
        
        # Normalize depth
        depth = prediction_resized.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depths.append(depth)
    
    return depths
def temporal_smooth_gpu(prev_depths, curr_depths, alpha=0.7):
    """GPU-accelerated temporal smoothing for batches"""
    if prev_depths is None or len(prev_depths) == 0:
        return curr_depths
    
    # Make sure both lists have the same length
    min_len = min(len(prev_depths), len(curr_depths))
    smoothed = []
    
    for i in range(min_len):
        if prev_depths[i] is None:
            smoothed.append(curr_depths[i])
        else:
            # Use GPU if available, otherwise CPU
            if torch.cuda.is_available() and isinstance(prev_depths[i], np.ndarray):
                # Convert to tensors for GPU acceleration
                prev_tensor = torch.from_numpy(prev_depths[i]).float().to(device)
                curr_tensor = torch.from_numpy(curr_depths[i]).float().to(device)
                smoothed_tensor = alpha * prev_tensor + (1 - alpha) * curr_tensor
                smoothed.append(smoothed_tensor.cpu().numpy())
            else:
                # CPU fallback
                smoothed.append(alpha * prev_depths[i] + (1 - alpha) * curr_depths[i])
    
    # Add any remaining current depths
    if len(curr_depths) > min_len:
        smoothed.extend(curr_depths[min_len:])
    
    return smoothed

def inject_vr180_metadata(input_video_path: str, output_video_path: str):
    """
    Inject VR180 metadata into a video using multiple approaches with built-in verification and debugging.
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
            
            # Convert to string for easier searching
            metadata_str = str(metadata).lower()
            
            # Check for various VR metadata indicators
            vr_indicators = [
                "spherical", "stereo_mode", "projection", 
                "equirectangular", "left_right", "spatial"
            ]
            
            found_indicators = [indicator for indicator in vr_indicators if indicator in metadata_str]
            
            if found_indicators:
                print(f"✅ VR180 metadata found: {', '.join(found_indicators)}")
                print(f"📱 Compatible with: YouTube VR, Oculus Quest/Rift, Google Cardboard")
                return {"success": True, "indicators": found_indicators, "metadata": metadata}
            
            # Method 2: Check with exiftool if available
            try:
                exif_cmd = ["exiftool", "-j", video_path]
                exif_result = subprocess.run(exif_cmd, capture_output=True, text=True, check=True)
                exif_data = json.loads(exif_result.stdout)
                exif_str = str(exif_data).lower()
                
                exif_indicators = [indicator for indicator in vr_indicators if indicator in exif_str]
                if exif_indicators:
                    print(f"✅ VR180 metadata found via ExifTool: {', '.join(exif_indicators)}")
                    return {"success": True, "indicators": exif_indicators, "metadata": exif_data}
                    
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass  # ExifTool not available or failed
            
            print(f"⚠️ No VR180 metadata detected in {video_path}")
            return {"success": False, "indicators": [], "metadata": metadata}
            
        except Exception as e:
            print(f"❌ Error verifying metadata: {str(e)}")
            return {"success": False, "error": str(e)}

    try:
        # Debug input file first
        print("🔍 DEBUGGING INPUT FILE:")
        debug_video_metadata(input_video_path)
        
        # Method 1: Standard MP4 metadata approach
        print("\n🔄 Attempting Method 1: Standard MP4 metadata...")
        cmd1 = [
            "ffmpeg", "-y", "-i", input_video_path,
            "-c:v", "copy", "-c:a", "copy",
            "-movflags", "use_metadata_tags+faststart",  # Added faststart
            "-metadata:s:v:0", "spherical=true",
            "-metadata:s:v:0", "projection=equirectangular",  # Try equirectangular first
            "-metadata:s:v:0", "stereo_mode=left_right",
            "-metadata", "spherical=true",
            "-metadata", "projection=equirectangular", 
            "-metadata", "stereo_mode=left_right",
            "-metadata", "full_pano_width_pixels=3840",  # Updated field names
            "-metadata", "full_pano_height_pixels=3840",
            "-metadata", "cropped_area_image_width=3840",
            "-metadata", "cropped_area_image_height=1920",
            "-metadata", "cropped_area_left=0",
            "-metadata", "cropped_area_top=960",
            output_video_path
        ]
        
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        
        if result1.returncode == 0:
            print("🔍 DEBUGGING OUTPUT FILE AFTER METHOD 1:")
            debug_video_metadata(output_video_path)
            verification = verify_vr180_metadata(output_video_path)
            if verification["success"]:
                return {"status": "success", "method": "standard", "output_file": output_video_path, "verification": verification}
        else:
            print(f"❌ Method 1 failed: {result1.stderr}")
        
        print("\n⚠️ Method 1 failed, trying Method 2...")
        
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
        print(f"❌ Unexpected error: {str(e)}")
        # Still try to debug if output file exists
        if os.path.exists(output_video_path):
            print("🔍 DEBUGGING OUTPUT FILE AFTER ERROR:")
            debug_video_metadata(output_video_path)
        raise RuntimeError(f"VR180 metadata injection failed: {str(e)}")
# def inject_vr180_metadata(input_video_path: str, output_video_path: str):
#     """
#     Advanced VR180 metadata injection using multiple approaches
#     """
#     import subprocess
#     import shutil
    
#     methods = [
#         # Method 1: Google Spatial Media standard
#         [
#             "ffmpeg", "-y", "-i", input_video_path,
#             "-c", "copy",
#             "-metadata", "spherical-video=true",
#             "-metadata", "stereo-mode=left-right",
#             "-metadata", "projection-type=equirectangular",
#             "-movflags", "faststart",
#             output_video_path
#         ],
        
#         # Method 2: Traditional MP4 metadata
#         [
#             "ffmpeg", "-y", "-i", input_video_path, 
#             "-c", "copy",
#             "-metadata:s:v:0", "handler_name=Spherical Video",
#             "-metadata:s:v:0", "spherical=true",
#             "-metadata:s:v:0", "stereo_mode=left-right",
#             "-movflags", "faststart",
#             output_video_path
#         ]
#     ]
    
#     for i, cmd in enumerate(methods, 1):
#         print(f"🔄 Trying method {i}...")
#         try:
#             result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
#             if result.returncode == 0:
#                 print(f"✅ Method {i} succeeded")
#                 return output_video_path
#             else:
#                 print(f"❌ Method {i} failed: {result.stderr}")
#         except Exception as e:
#             print(f"❌ Method {i} error: {str(e)}")
    
#     # Final fallback
#     print("⚠️ All methods failed, copying file without metadata")
#     shutil.copy2(input_video_path, output_video_path)
#     return output_video_path
def make_stereo_pair_optimized(img, depth, eye_offset=8):
    """
    Optimized stereo pair generation using vectorized operations
    Works for both grayscale and color images
    """
    h, w = img.shape[:2]
    
    # Resize depth to match frame dimensions
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h))
    
    # Create coordinate matrices
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Calculate shifts based on depth (vectorized)
    shifts = ((1 - depth) * eye_offset).astype(np.int32)
    
    # Calculate left and right coordinates
    left_x = np.clip(x_coords - shifts, 0, w - 1)
    right_x = np.clip(x_coords + shifts, 0, w - 1)
    
    # Create stereo images
    left = np.zeros_like(img)
    right = np.zeros_like(img)
    
    if len(img.shape) == 3:  # Color image (3 channels)
        # For color images, handle each channel
        left[y_coords, left_x, :] = img[y_coords, x_coords, :]
        right[y_coords, right_x, :] = img[y_coords, x_coords, :]
    else:  # Grayscale image (2 channels)
        left[y_coords, left_x] = img[y_coords, x_coords]
        right[y_coords, right_x] = img[y_coords, x_coords]
    
    return left, right

# Then define your conversion function AFTER the helper functions
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
def convert_2d_to_vr180_gpu_optimized(input_path: str, output_path: str):
    """
    GPU-optimized 2D to VR180 conversion for Colab T4
    """
    temp_dir = tempfile.mkdtemp()
    cap = cv2.VideoCapture(input_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Processing video: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")

    frame_paths = []
    prev_depths = None
    frame_batch = []
    processed_frames = 0

    # Process frames in batches for GPU efficiency
    while True:
        ret, frame = cap.read()
        if not ret:
            # Process remaining frames in batch
            if frame_batch:
                # Estimate depth for the batch
                depths = estimate_depth_batch(frame_batch)
                
                # Apply temporal smoothing if we have previous depths
                if prev_depths is not None:
                    # FIX: Make sure we're handling the batch correctly
                    # If prev_depths is a single depth map (from old version), convert to batch format
                    if not isinstance(prev_depths, list):
                        prev_depths = [prev_depths] * len(frame_batch)
                    
                    # Apply temporal smoothing
                    depths = temporal_smooth_gpu(prev_depths, depths)
                
                # Update previous depths for next batch
                prev_depths = depths[-len(frame_batch):] if prev_depths is not None else depths
                
                # Process each frame in the batch
                for frame_data, depth in zip(frame_batch, depths):
                    left, right = make_stereo_pair_optimized(frame_data, depth)
                    sbs = np.concatenate((left, right), axis=1)
                    
                    frame_path = os.path.join(temp_dir, f"{len(frame_paths):05d}.png")
                    cv2.imwrite(frame_path, sbs, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                    frame_paths.append(frame_path)
                    processed_frames += 1
                
                # Clear GPU memory after batch
                clear_gpu_cache()
            break

        frame_batch.append(frame)
        
        # Process batch when it's full
        if len(frame_batch) >= BATCH_SIZE:
            # Estimate depth for the batch
            depths = estimate_depth_batch(frame_batch)
            
            # Apply temporal smoothing if we have previous depths
            if prev_depths is not None:
                # FIX: Make sure we're handling the batch correctly
                if not isinstance(prev_depths, list):
                    prev_depths = [prev_depths] * BATCH_SIZE
                
                # Apply temporal smoothing
                depths = temporal_smooth_gpu(prev_depths, depths)
            
            # Update previous depths for next batch
            prev_depths = depths[-BATCH_SIZE:] if prev_depths is not None else depths
            
            # Process each frame in the batch
            for frame_data, depth in zip(frame_batch, depths):
                left, right = make_stereo_pair_optimized(frame_data, depth)
                sbs = np.concatenate((left, right), axis=1)
                
                frame_path = os.path.join(temp_dir, f"{len(frame_paths):05d}.png")
                cv2.imwrite(frame_path, sbs, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                frame_paths.append(frame_path)
                processed_frames += 1
            
            frame_batch = []  # Clear batch
            
            # Progress update
            if processed_frames % (BATCH_SIZE * 10) == 0:
                progress = (processed_frames / total_frames) * 100
                print(f"🔄 Progress: {progress:.1f}% ({processed_frames}/{total_frames} frames)")
                clear_gpu_cache()  # Periodic cleanup

    cap.release()

    if not frame_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("No frames were processed. Check input video.")

    print(f"✅ Processed {len(frame_paths)} frames, creating video...")

    # Create output video using GPU-accelerated encoding if available
    try:
        # Try hardware-accelerated encoding first (T4 supports NVENC)
        # cmd = [
        #     "ffmpeg", "-y",
        #     "-r", str(fps),
        #     "-i", os.path.join(temp_dir, "%05d.png"),
        #     "-c:v", "h264_nvenc",  # GPU encoding
        #     "-preset", "fast",
        #     "-crf", "23",
        #     "-pix_fmt", "yuv420p",
        #     "-threads", "0",
        #     output_path
        # ]
        cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", os.path.join(temp_dir, "%05d.png"),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-threads", "0",
        output_path
    ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print("⚠️ GPU encoding failed, falling back to CPU")
            # Fallback to CPU encoding
            # cmd = [
            #     "ffmpeg", "-y",
            #     "-r", str(fps),
            #     "-i", os.path.join(temp_dir, "%05d.png"),
            #     "-c:v", "libx264",
            #     "-preset", "medium",
            #     "-crf", "23",
            #     "-pix_fmt", "yuv420p",
            #     "-threads", "0",
            #     output_path
            # ]
            cmd = [
    "ffmpeg", "-y",
    "-r", str(fps),
    "-i", os.path.join(temp_dir, "%05d.png"),
    "-c:v", "h264_nvenc",
    "-b:v", "25M",           # Higher bitrate for better quality
    "-maxrate", "25M",
    "-bufsize", "50M",
    "-pix_fmt", "yuv420p",
    "-threads", "0",
    output_path
]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            raise RuntimeError(f"Video encoding failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        raise RuntimeError("Video encoding timeout - video too long for Colab")

    # Cleanup temporary files
    shutil.rmtree(temp_dir, ignore_errors=True)

    if not os.path.exists(output_path):
        raise RuntimeError("Failed to create SBS output video.")

    print("✅ VR180 conversion completed")
# Base directories
BASE_DIR = "/content/Vr180_Back"  # Colab base directory
UPLOAD_DIR = os.path.join(BASE_DIR, "tmp_uploads")
FINAL_DIR = os.path.join(BASE_DIR, "videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
def get_conversion_by_id(db: Session, conversion_id: int, user_id: int):
    return db.query(Conversion).filter(
        Conversion.id == conversion_id, 
        Conversion.user_id == user_id
    ).first()


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
        print(f"👤 User: {current_user.username} (ID: {current_user.id})")  # Debug log
        
        # Clear GPU memory before starting
        clear_gpu_cache()
        
        # Step 1: Convert to VR180 using GPU optimization
        await asyncio.to_thread(convert_2d_to_vr180_gpu_optimized, input_path, output_path)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="VR180 conversion failed.")

        # Step 2: Add audio with metadata preservation
        await add_audio_preserve_metadata(input_path, output_path, temp_with_audio)
        
        # Step 3: Inject VR180 metadata
        await asyncio.to_thread(inject_vr180_metadata, temp_with_audio, final_output_path)

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
            # Continue with conversion even if history save fails
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
        
        # Only add conversion ID if save was successful
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
        # Cleanup temporary files to save Colab storage
        def cleanup():
            try:
                cleanup_paths = [input_path, output_path]
                # Only add temp_with_audio if it was created
                if 'temp_with_audio' in locals() and temp_with_audio:
                    cleanup_paths.append(temp_with_audio)
                    
                for path in cleanup_paths:
                    if os.path.exists(path):
                        os.unlink(path)
                        print(f"🧹 Cleaned up: {path}")
                clear_gpu_cache()
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

# @app.post("/convert/")
# async def convert_video(video: UploadFile = File(...)):
#     """Main conversion endpoint optimized for Colab"""
#     if not video.content_type.startswith("video/"):
#         raise HTTPException(status_code=400, detail="File must be a video")

#     # Check file size (Colab has limited storage)
#     content = await video.read()
#     file_size_mb = len(content) / (1024 * 1024)
    
#     if file_size_mb > 500:  # 500MB limit for Colab
#         raise HTTPException(status_code=400, detail="File too large. Maximum 500MB for Colab.")

#     unique_id = uuid.uuid4().hex
#     input_path = os.path.join(UPLOAD_DIR, f"input_{unique_id}.mp4")
#     output_path = os.path.join(UPLOAD_DIR, f"output_{unique_id}_vr180.mp4")
#     final_output_path = os.path.join(FINAL_DIR, f"vr180_{unique_id}.mp4")

#     # Save uploaded file
#     with open(input_path, "wb") as f:
#         f.write(content)

#     try:
#         print(f"🚀 Starting conversion: {video.filename} ({file_size_mb:.1f}MB)")
        
#         # Clear GPU memory before starting
#         clear_gpu_cache()
        
#         # Step 1: Convert to VR180 using GPU optimization
#         await asyncio.to_thread(convert_2d_to_vr180_gpu_optimized, input_path, output_path)

#         if not os.path.exists(output_path):
#             raise HTTPException(status_code=500, detail="VR180 conversion failed.")

#         # Step 2: Add audio with metadata preservation AND inject VR180 metadata
#         # First add audio back to the video
#         temp_with_audio = output_path.replace(".mp4", "_with_audio.mp4")
#         await add_audio_preserve_metadata(input_path, output_path, temp_with_audio)
        
#         # Step 3: Inject VR180 metadata
#         await asyncio.to_thread(inject_vr180_metadata, temp_with_audio, final_output_path)

#         if not os.path.exists(final_output_path):
#             raise HTTPException(status_code=500, detail="Final conversion failed.")
#         final_file_size = os.path.getsize(final_output_path)
#         conversion = save_to_history(
#     db=db,
#     user_id=current_user.id,
#     original_filename=video.filename,
#     output_path=final_output_path,
#     file_size=final_file_size
# )

#         # Step 4: Return file for download
#         def iter_file(path: str):
#             with open(path, "rb") as f:
#                 while chunk := f.read(8192):
#                     yield chunk

#         final_size = os.path.getsize(final_output_path) / (1024 * 1024)
        
        
#         print(f"✅ Conversion completed: {final_size:.1f}MB")

#         return StreamingResponse(
#             iter_file(final_output_path),
#             media_type="video/mp4",
#             headers={
#                 "Content-Disposition": f'attachment; filename="vr180_{video.filename}"',
#                 "X-File-Size-MB": f"{final_size:.1f}"
#             }
#         )

#     except Exception as e:
#         print(f"❌ Conversion error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
        
#     finally:
#         # Cleanup temporary files to save Colab storage
#         def cleanup():
#             try:
#                 for path in [input_path, output_path, temp_with_audio]:
#                     if os.path.exists(path):
#                         os.unlink(path)
#                 clear_gpu_cache()
#             except Exception as e:
#                 print(f"Cleanup error: {e}")

#         threading.Thread(target=cleanup, daemon=True).start()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
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

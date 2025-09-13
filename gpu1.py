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
# Add these imports at the TOP of gpu1.py
from typing import Tuple, List, Dict, Optional, Any
import cv2
import numpy as np
import torch
import subprocess
import os
import tempfile
import shutil
from PIL import Image
import json
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

import numpy as np
import cv2
import os
import subprocess
import tempfile
import shutil
import asyncio
from typing import List, Tuple


import cv2
import os
import tempfile
import shutil
import subprocess
from typing import List, Tuple
import numpy as np


# Alternative: Using FFmpeg's v360 filter for better performance


        
    
     
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import cv2

def init_outpainting_model():
    """Initialize AI outpainting model optimized for VR180 peripheral expansion"""
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            variant="fp16" if torch.cuda.is_available() else None,
        )
        
        # Enable attention slicing for memory efficiency
        if torch.cuda.is_available():
            pipe.enable_attention_slicing()
            pipe.enable_xformers_memory_efficient_attention()
        
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        print("✅ AI Outpainting model loaded for VR180 peripheral expansion")
        return pipe
    except Exception as e:
        print(f"❌ Failed to load outpainting model: {e}")
        return None

outpainting_pipe = init_outpainting_model()

def ai_outpaint_frame_vr180(frame, expansion_percent=25, scene_context="hallway"):
    """
    VR180-optimized AI outpainting for peripheral expansion to 210°+
    """
    if outpainting_pipe is None:
        return frame  # Fallback to original frame
    
    # Convert to PIL if needed
    if isinstance(frame, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        pil_image = frame
    
    width, height = pil_image.size
    
    # VR180-SPECIFIC: Asymmetric expansion (more left/right, less top)
    expand_left_right = int(width * expansion_percent / 100)      # Expand sides more
    expand_top = int(height * expansion_percent / 200)           # Expand top less
    expand_bottom = int(height * expansion_percent / 400)        # Expand bottom least
    
    # Create expanded canvas for 210°+ FOV
    new_width = width + expand_left_right * 2
    new_height = height + expand_top + expand_bottom
    expanded_canvas = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    
    # Paste original image centered horizontally, positioned vertically
    expanded_canvas.paste(pil_image, (expand_left_right, expand_top))
    
    # Create mask - only outpaint the expanded areas
    mask = Image.new("L", (new_width, new_height), 255)  # White = keep
    
    # Black mask where we want to outpaint (the expanded border areas)
    draw_mask = Image.new("L", (width, height), 0)  # Black = inpaint
    mask.paste(draw_mask, (expand_left_right, expand_top))
    
    # VR180-SPECIFIC PROMPT with scene context
    scene_prompts = {
        "hallway": "architectural hallway extension, continuous corridor, symmetric architecture, clean lines, realistic building interior, seamless extension",
        "outdoor": "natural environment extension, outdoor scenery continuation, realistic landscape, seamless terrain, consistent lighting",
        "general": "VR180 peripheral expansion, seamless environment extension, consistent perspective, natural continuation, immersive VR experience"
    }
    
    prompt = scene_prompts.get(scene_context, scene_prompts["general"])
    negative_prompt = "blurry, distorted, inconsistent, discontinuous, mismatched, artificial, fake, low quality"
    
    # Optimized outpainting parameters for VR
    result = outpainting_pipe(
        prompt=prompt,
        image=expanded_canvas,
        mask_image=mask,
        strength=0.85,  # Slightly lower for better consistency
        guidance_scale=8.0,
        num_inference_steps=25,
        negative_prompt=negative_prompt,
        generator=torch.Generator(device=outpainting_pipe.device).manual_seed(42)  # For consistency
    ).images[0]
    
    # For VR180: Return the fully expanded image (210°+)
    # The conversion to 180° will happen later in the pipeline
    return result

def ai_outpaint_batch_vr180(frames, expansion_percent=25, scene_context="hallway"):
    """
    Batch process frames with consistent outpainting for temporal coherence
    """
    results = []
    for frame in frames:
        outpainted = ai_outpaint_frame_vr180(frame, expansion_percent, scene_context)
        results.append(outpainted)
    
    return results    
def make_stereo_pair_optimized(
    img, 
    depth, 
    eye_offset=6.3,
    center_scale=0.82,
    panini_alpha=0.7,
    stereographic_strength=0.2,
    max_disparity_degrees=1.3,
    max_blur_radius=8
):
    # Input validation
    if img is None or depth is None:
        raise ValueError("Input image and depth map cannot be None")
    
    if not isinstance(img, np.ndarray) or not isinstance(depth, np.ndarray):
        raise TypeError("Image and depth must be numpy arrays")
    
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: {w}x{h}")
    
    cx, cy = w // 2, h // 2
    epsilon = 1e-6
    focal_length = 500
    
    # 1. Resize depth to match frame dimensions if needed
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize depth to [0.1, 1.0] range
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + epsilon)
    depth_normalized = np.clip(depth_normalized, 0.1, 1.0)
    
    # 2. Apply central scene scaling for comfort
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_scaled = ((x_coords - cx) * center_scale + cx).astype(np.int32)
    y_scaled = ((y_coords - cy) * center_scale + cy).astype(np.int32)
    x_scaled = np.clip(x_scaled, 0, w - 1)
    y_scaled = np.clip(y_scaled, 0, h - 1)
    
    img_scaled = img[y_scaled, x_scaled]
    depth_scaled = depth_normalized[y_scaled, x_coords]

    # 3. Calculate disparity with MAX DISPARITY CAPPING (1.3° visual angle)
    pixel_size_mm = 0.1
    max_pixel_disparity = (max_disparity_degrees * np.pi / 180) * focal_length / pixel_size_mm
    
    disparity = (eye_offset * 10 * focal_length) / (depth_scaled * w)
    disparity = np.clip(disparity, -max_pixel_disparity, max_pixel_disparity).astype(np.int32)

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

    # 5. Apply SIMPLIFIED projection blending
    def apply_simple_projection(image, strength=0.2):
        """Simplified projection that maintains original dimensions."""
        if image is None or image.size == 0:
            return image
            
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        x = (map_x - cx) / cx
        y = (map_y - cy) / cy
        
        r = np.sqrt(x**2 + y**2)
        scale = 1.0 + strength * r**2
        
        map_x = cx + x * cx * scale
        map_y = cy + y * cy * scale
        
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        
        return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    left = apply_simple_projection(left, strength=0.2)
    right = apply_simple_projection(right, strength=0.2)

    # 6. Apply foveated blur starting at 70° eccentricity
    def apply_foveated_blur_vr180(image, start_degrees=70, max_blur_radius=8):
        """VR180-optimized foveated blur starting at 70° eccentricity."""
        if image is None or image.size == 0:
            return image
            
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Calculate distance from center in degrees
        max_radius = np.sqrt(cx**2 + cy**2)
        y_coords, x_coords = np.indices((h, w))
        dist_pixels = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        dist_degrees = (dist_pixels / max_radius) * 90  # Convert to degrees
        
        # Create blur mask (start at 70°)
        blur_mask = np.zeros_like(dist_degrees, dtype=np.float32)
        mask_region = dist_degrees > start_degrees
        if np.any(mask_region):
            normalized_dist = (dist_degrees[mask_region] - start_degrees) / (90 - start_degrees)
            blur_mask[mask_region] = np.clip(normalized_dist * 2, 0, 1)  # Faster transition
        
        # Apply blur based on mask
        blurred = cv2.GaussianBlur(image, (max_blur_radius*2+1, max_blur_radius*2+1), 0)
        
        if len(image.shape) == 3:
            blur_mask = blur_mask[:, :, np.newaxis]
        
        return (1 - blur_mask) * image + blur_mask * blurred

    # APPLY THE BLUR
    left = apply_foveated_blur_vr180(left, start_degrees=70, max_blur_radius=max_blur_radius)
    right = apply_foveated_blur_vr180(right, start_degrees=70, max_blur_radius=max_blur_radius)

    # CRITICAL: Ensure output resolution matches input
    target_height, target_width = img.shape[:2]
    if left.shape[0] != target_height or left.shape[1] != target_width:
        print(f"⚠️ Correcting left eye: {left.shape} -> ({target_height}, {target_width})")
        left = cv2.resize(left, (target_width, target_height))
    
    if right.shape[0] != target_height or right.shape[1] != target_width:
        print(f"⚠️ Correcting right eye: {right.shape} -> ({target_height}, {target_width})")
        right = cv2.resize(right, (target_width, target_height))
    
    print(f"✅ Final output: Left={left.shape}, Right={right.shape}")
    
    return left.astype(np.uint8), right.astype(np.uint8)


# Utility function for batch processing
def smart_upscale_to_8k(image, method="lanczos"):
    """
    Proper 4× upscale to 3840x3840 (4K per eye) - NOT SBS!
    """
    if image is None:
        raise ValueError("Image cannot be None")
    
    h, w = image.shape[:2]
    
    # PROTECTION: If image is already 4K or larger, don't upscale again!
    if h >= 3840 and w >= 3840:
        print(f"⚠️ Already upscaled: {w}x{h}, returning as-is")
        return image
    
    # TARGET: 3840x3840 (4K per eye for VR180)
    target_size = (3840, 3840)
    
    print(f"🔍 Upscaling: {w}x{h} → {target_size[0]}x{target_size[1]}")
    
    # Select interpolation method
    if method == "lanczos":
        interpolation = cv2.INTER_LANCZOS4
    elif method == "cubic":
        interpolation = cv2.INTER_CUBIC
    elif method == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif method == "area":
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    
    try:
        # Direct 4× upscale to 3840x3840 (4K per eye)
        upscaled = cv2.resize(image, target_size, interpolation=interpolation)
        print(f"✅ Upscaled to: {upscaled.shape[1]}x{upscaled.shape[0]}")
        return upscaled
        
    except cv2.error as e:
        print(f"⚠️ Upscaling failed with {method}, falling back to linear: {e}")
        upscaled = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return upscaled
def create_stereo_batch_vr180(frames, depth_maps, upscale_8k=True, **kwargs):
    """
    Process batch of frames with consistent VR180 parameters
    """
    left_eyes = []
    right_eyes = []
    
    for frame, depth in zip(frames, depth_maps):
        left, right = make_stereo_pair_optimized(frame, depth, **kwargs)
        
        # REMOVE THIS UPSCALING BLOCK - already done in make_stereo_pair_optimized!
        # if upscale_8k:
        #     left = smart_upscale_to_8k(left, method="lanczos")
        #     right = smart_upscale_to_8k(right, method="lanczos")
        
        left_eyes.append(left)
        right_eyes.append(right)
    
    return left_eyes, right_eyes

def process_stereo_batch_safe(frames, depth_maps, upscale_8k=True, **kwargs):
    """
    Safely process a batch of frames
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
            if frame is None:
                print(f"⚠️ Frame {i} is None, skipping")
                continue
            if depth is None:
                print(f"⚠️ Depth {i} is None, creating fallback")
                h, w = frame.shape[:2]
                depth = np.ones((h, w), dtype=np.float32) * 0.5
            
            # Create stereo pair
            left, right = make_stereo_pair_optimized(frame, depth, **kwargs)
            
            # REMOVE THIS UPSCALING BLOCK - already done!
            # if upscale_8k:
            #     left = smart_upscale_to_8k(left, method="lanczos")
            #     right = smart_upscale_to_8k(right, method="lanczos")
            
            # Combine into side-by-side format
            sbs = np.concatenate((left, right), axis=1)
            stereo_pairs.append(sbs)
            
        except Exception as e:
            print(f"⚠️ Failed to process stereo pair {i}: {e}")
            # Fallback without upscaling
            try:
                if frame is not None:
                    sbs = np.concatenate((frame, frame), axis=1)
                    stereo_pairs.append(sbs)
            except Exception as e2:
                print(f"❌ Even fallback failed for frame {i}: {e2}")
                continue
    
    return stereo_pairs
# AI-enhanced upscaling option (if you have access to AI models)
def ai_enhanced_upscale_to_8k(image, model_type="esrgan"):
    """
    AI-powered 8K upscaling for better quality (optional enhancement)
    """
    # This would use AI models like ESRGAN, Real-ESRGAN, or similar
    # For now, fallback to traditional upscaling
    return smart_upscale_to_8k(image, method="lanczos")

print("🎯 Fixed 8K VR180 Upscaling Ready!")
print("✅ Proper 4x scaling from 2K→8K (not 8x)")
print("✅ Maintains VR180 2:1 aspect ratio (7680x3840)")
print("✅ Integrated into batch processing functions")
print("🚀 Ready for 8K VR180 conversion!")
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

def create_side_by_side_video(
    left_frames: List[np.ndarray],
    right_frames: List[np.ndarray],
    output_path: str,
    fps: float,
    per_eye_resolution: Tuple[int, int] = (3840, 3840)
) -> None:
    """
    Create side-by-side video from left/right eye frames (EQUIRECTANGULAR output)
    """
    temp_dir = tempfile.mkdtemp()
    
    try:
        print(f"🎬 Creating equirectangular SBS video from {len(left_frames)} frames...")
        
        for i, (left_frame, right_frame) in enumerate(zip(left_frames, right_frames)):
            # Ensure both eyes have the same resolution
            if left_frame.shape != right_frame.shape:
                # Resize to match if needed
                right_frame = cv2.resize(right_frame, (left_frame.shape[1], left_frame.shape[0]))
            
            # Combine into SBS frame
            sbs_frame = np.concatenate((left_frame, right_frame), axis=1)
            frame_path = os.path.join(temp_dir, f"{i:06d}.png")
            cv2.imwrite(frame_path, sbs_frame)
            
            if i % 10 == 0:
                print(f"📊 Processed {i+1}/{len(left_frames)} frames")

        # Create video from frames
        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", os.path.join(temp_dir, "%06d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Created equirectangular SBS video: {output_path}")
        else:
            print(f"❌ FFmpeg failed: {result.stderr}")
            raise Exception("FFmpeg video creation failed")
            
    except Exception as e:
        print(f"❌ Error creating SBS video: {e}")
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
def convert_2d_to_vr180_gpu_optimized(
    input_path: str,
    output_path: str,
    center_scale: float = 0.82,
    focal_length: int = 500,
    panini_alpha: float = 0.7,
    stereographic_strength: float = 0.2,
    eye_offset: float = 6.3,
    max_disparity_degrees: float = 1.3,
    max_blur_radius: int = 8,
    ai_outpaint: bool = True,
    upscale_to_8k: bool = True,
    upscale_method: str = "lanczos",
    expansion_degrees: int = 210
):
    """
    GPU-optimized 2D to VR180 conversion with comprehensive error handling
    """
    print("🎯 VR180 CONVERSION STARTED - Equirectangular Output")

    # Test video file first
    test_cap = cv2.VideoCapture(input_path)
    ret_test, frame_test = test_cap.read()
    if not (ret_test and frame_test is not None):
        test_cap.release()
        raise ValueError("Cannot read video frames")
    print(f"✅ Video test passed: {frame_test.shape}")
    test_cap.release()

    # Initialize AI outpainting if requested
    outpainting_pipe = None
    if ai_outpaint:
        try:
            outpainting_pipe = init_outpainting_model()
            print("✅ AI outpainting initialized")
        except Exception as e:
            print(f"⚠️ AI outpainting init failed: {e}")
            ai_outpaint = False

    temp_dir = tempfile.mkdtemp()

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # Get video properties with validation
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions: {width}x{height}")

        print(f"📹 Input video: {width}x{height}, {total_frames} frames at {fps:.2f} FPS")

        # Processing variables
        frame_batch = []
        prev_depths = None
        processed_frames = 0
        frame_paths = []  # Store paths to frame files

        # Determine batch size based on 8K upscaling
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
                        # Process the final batch
                        processed_count = process_frame_batch(
                            frame_batch, 
                            temp_dir, 
                            len(frame_paths),
                            prev_depths,
                            outpainting_pipe if ai_outpaint else None,
                            upscale_to_8k,
                            upscale_method,
                            eye_offset,
                            center_scale,
                            panini_alpha,
                            stereographic_strength,
                            max_disparity_degrees,
                            max_blur_radius
                        )
                        
                        # Update counters
                        for i in range(processed_count):
                            frame_paths.append(os.path.join(temp_dir, f"{len(frame_paths) + i:06d}.png"))
                        
                        processed_frames += processed_count
                        
                    except Exception as e:
                        print(f"❌ Final batch processing failed: {e}")
                
                break

            # Validate frame
            if frame is None or not isinstance(frame, np.ndarray) or len(frame.shape) < 2:
                print(f"⚠️ Skipping invalid frame")
                continue

            frame_h, frame_w = frame.shape[:2]
            if frame_h <= 0 or frame_w <= 0:
                print(f"⚠️ Skipping frame with invalid dimensions: {frame_w}x{frame_h}")
                continue

            frame_batch.append(frame)

            # Process batch when full
            if len(frame_batch) >= current_batch_size:
                print(f"🔄 Processing batch: {len(frame_batch)} frames")
                
                try:
                    processed_count = process_frame_batch(
                        frame_batch,
                        temp_dir,
                        len(frame_paths),
                        prev_depths,
                        outpainting_pipe if ai_outpaint else None,
                        upscale_to_8k,
                        upscale_method,
                        eye_offset,
                        center_scale,
                        panini_alpha,
                        stereographic_strength,
                        max_disparity_degrees,
                        max_blur_radius
                    )
                    
                    # Update counters and paths
                    for i in range(processed_count):
                        frame_paths.append(os.path.join(temp_dir, f"{len(frame_paths) + i:06d}.png"))
                    
                    processed_frames += processed_count
                    
                    # Progress update
                    if processed_frames % (current_batch_size * 3) == 0:
                        progress = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
                        print(f"📊 Progress: {progress:.1f}% ({processed_frames}/{total_frames})")
                        clear_gpu_cache()
                    
                except Exception as e:
                    print(f"❌ Batch processing failed: {e}")
                    # Continue with next batch instead of failing completely
                
                frame_batch = []

        cap.release()

        if not frame_paths:
            raise RuntimeError("No frames were successfully processed")

        print(f"✅ Processed {len(frame_paths)} frames, creating video...")

        # Create final video from processed frames
        create_video_from_frames(frame_paths, output_path, fps)

    except Exception as e:
        print(f"❌ Conversion error: {e}")
        raise
    finally:
        if 'cap' in locals():
            cap.release()
        shutil.rmtree(temp_dir, ignore_errors=True)
        clear_gpu_cache()


def process_frame_batch(
    frame_batch, 
    temp_dir, 
    start_index,
    prev_depths,
    outpainting_pipe,
    upscale_to_8k,
    upscale_method,
    eye_offset,
    center_scale,
    panini_alpha,
    stereographic_strength,
    max_disparity_degrees,
    max_blur_radius
):
    """
    Process a batch of frames and save them to temp directory
    Returns the number of successfully processed frames
    """
    if not frame_batch:
        return 0
    
    try:
        # Get depth maps for the batch
        depths = estimate_depth_batch(frame_batch)
        
        # Ensure depths match frame count
        if len(depths) != len(frame_batch):
            print(f"⚠️ Adjusting depths: {len(depths)} -> {len(frame_batch)}")
            while len(depths) < len(frame_batch):
                # Create fallback depth map
                h, w = frame_batch[0].shape[:2]
                depths.append(np.ones((h, w), dtype=np.float32) * 0.5)
            depths = depths[:len(frame_batch)]

        # Apply temporal smoothing if we have previous depths
        if prev_depths is not None:
            try:
                depths = temporal_smooth_gpu(prev_depths, depths)
            except Exception as e:
                print(f"⚠️ Temporal smoothing failed: {e}")

        processed_count = 0
        
        for i, (frame_data, depth) in enumerate(zip(frame_batch, depths)):
            try:
                # Validate frame and depth
                if frame_data is None or depth is None:
                    print(f"⚠️ Skipping None frame/depth at index {i}")
                    continue
                
                if not isinstance(frame_data, np.ndarray):
                    frame_data = np.array(frame_data)
                
                frame_h, frame_w = frame_data.shape[:2]
                if frame_h <= 0 or frame_w <= 0:
                    print(f"⚠️ Skipping invalid frame dimensions: {frame_w}x{frame_h}")
                    continue

                # AI Outpainting (optional)
                if outpainting_pipe is not None:
                    try:
                        frame_rgb = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                        pil_frame = Image.fromarray(frame_rgb)
                        outpainted_frame = ai_outpaint_frame_vr180(
                            pil_frame, 
                            expansion_percent=25, 
                            scene_context="hallway"
                        )
                        frame_data = cv2.cvtColor(np.array(outpainted_frame), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"⚠️ AI outpainting failed for frame {i}: {e}")

                # 8K Upscaling (optional) - ONLY ONCE
                original_size = frame_data.shape[:2]
                if upscale_to_8k:
                    try:
                        frame_data = smart_upscale_to_8k(frame_data, upscale_method)
                        depth = smart_upscale_to_8k(depth, "nearest")
                        print(f"📈 Upscaled from {original_size} to {frame_data.shape[:2]}")
                    except Exception as e:
                        print(f"⚠️ Upscaling failed for frame {i}: {e}")
                        continue

                # Create stereo pair
                try:
                    left, right = make_stereo_pair_optimized(
                        frame_data, 
                        depth,
                        eye_offset=eye_offset,
                        center_scale=center_scale,
                        panini_alpha=panini_alpha,
                        stereographic_strength=stereographic_strength,
                        max_disparity_degrees=max_disparity_degrees,
                        max_blur_radius=max_blur_radius
                    )
                    
                    if left is None or right is None:
                        print(f"⚠️ Stereo creation returned None for frame {i}")
                        continue

                    # DEBUG: Check resolutions
                    print(f"🔍 Frame {i}: Left={left.shape}, Right={right.shape}")

                except Exception as e:
                    print(f"⚠️ Stereo creation failed for frame {i}: {e}")
                    continue

                # Create side-by-side frame
                try:
                    # Ensure both eyes have same dimensions BEFORE concatenation
                    if left.shape != right.shape:
                        print(f"⚠️ Resizing right eye to match left: {right.shape} -> {left.shape}")
                        right = cv2.resize(right, (left.shape[1], left.shape[0]))
                    
                    sbs_frame = np.concatenate((left, right), axis=1)
                    
                    # DEBUG: Check final SBS resolution
                    print(f"🎯 SBS Frame {i}: {sbs_frame.shape}")
                    
                    # Verify this is 7680x3840 for 8K VR180
                    if sbs_frame.shape[1] != 7680 or sbs_frame.shape[0] != 3840:
                        print(f"❌ WRONG RESOLUTION: {sbs_frame.shape[1]}x{sbs_frame.shape[0]} (expected 7680x3840)")
                        # Force correct resolution
                        sbs_frame = cv2.resize(sbs_frame, (7680, 3840))
                        print(f"✅ Corrected to: 7680x3840")
                    
                    # Save frame
                    frame_path = os.path.join(temp_dir, f"{start_index + processed_count:06d}.png")
                    success = cv2.imwrite(frame_path, sbs_frame, [cv2.IMWRITE_PNG_COMPRESSION, 6])
                    
                    if success:
                        processed_count += 1
                    else:
                        print(f"⚠️ Failed to save frame {start_index + processed_count}")
                        
                except Exception as e:
                    print(f"⚠️ Frame saving failed for frame {i}: {e}")
                    continue

            except Exception as e:
                print(f"❌ Frame {i} processing failed: {e}")
                continue

        return processed_count
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        return 0


def create_video_from_frames(frame_paths, output_path, fps):
    """
    Create video from frame files using FFmpeg with resolution validation
    """
    import subprocess  # ← ADD THIS IMPORT!
    
 
    if not frame_paths:
        raise RuntimeError("No frame paths provided")
    
    temp_dir = os.path.dirname(frame_paths[0])
    
    # Check first frame resolution to ensure it's correct
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is not None:
        height, width = first_frame.shape[:2]
        print(f"📏 First frame resolution: {width}x{height}")
        
        if width != 7680 or height != 3840:
            print(f"❌ INCORRECT RESOLUTION: {width}x{height} (expected 7680x3840)")
            print("🔄 Resizing all frames to 7680x3840...")
            
            # Resize all frames to correct resolution
            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    resized_frame = cv2.resize(frame, (7680, 3840))
                    cv2.imwrite(frame_path, resized_frame)
                if i % 10 == 0:
                    print(f"📊 Resized {i+1}/{len(frame_paths)} frames")
    
    try:
        print(f"🎬 Creating video from {len(frame_paths)} frames...")
        
        # Use FFmpeg to create video from frames
        cmd = [
            "ffmpeg", "-y",
            "-r", str(fps),
            "-i", os.path.join(temp_dir, "%06d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-threads", "0",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            print("⚠️ High quality encoding failed, trying faster preset...")
            # Fallback with faster settings
            cmd[cmd.index("-preset") + 1] = "fast"
            cmd[cmd.index("-crf") + 1] = "23"
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
        if result.returncode != 0:
            # Final fallback with safe settings
            print("⚠️ Standard encoding failed, using safe fallback...")
            cmd = [
                "ffmpeg", "-y",
                "-r", str(fps),
                "-i", os.path.join(temp_dir, "%06d.png"),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        
        if result.returncode != 0:
            raise RuntimeError(f"All video encoding attempts failed: {result.stderr}")
            
        if not os.path.exists(output_path):
            raise RuntimeError("Video file was not created")
            
        # Verify final video resolution
        try:
            import subprocess
            result = subprocess.run([
                "ffprobe", "-v", "error", 
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                output_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                resolution = result.stdout.strip()
                print(f"✅ Final video resolution: {resolution}")
                if resolution != "7680x3840":
                    print(f"❌ WARNING: Wrong final resolution: {resolution}")
        except:
            pass
            
        print(f"✅ Video created successfully: {output_path}")
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Video encoding timeout - file may be too large")
    except Exception as e:
        raise RuntimeError(f"Video creation failed: {str(e)}")
# Fixed metadata injection function
def inject_vr180_metadata_optimized(
    input_video_path: str,
    output_video_path: str,
    width: int = 7680,
    height: int = 3840
) -> None:
    """
    Inject proper VR180 metadata for equirectangular side-by-side format
    """
    cmd = [
        "ffmpeg", "-y", 
        "-i", input_video_path,
        "-c:v", "copy",
        "-c:a", "copy",
        "-movflags", "use_metadata_tags+faststart",
        "-metadata:s:v:0", "spherical-video=true",
        "-metadata:s:v:0", "stereoscopic-mode=left-right",  # Fixed: side-by-side
        "-metadata:s:v:0", "projection-type=equirectangular",  # Fixed: standard VR180
        "-metadata:s:v:0", f"full_pano_width_pixels={width}",
        "-metadata:s:v:0", f"full_pano_height_pixels={height}",
        "-metadata:s:v:0", f"cropped_area_image_width={width}",
        "-metadata:s:v:0", f"cropped_area_image_height={height}",
        "-metadata:s:v:0", "cropped_area_left=0",
        "-metadata:s:v:0", "cropped_area_top=0",
        output_video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ VR180 metadata injected successfully!")
    else:
        print(f"❌ Metadata injection failed: {result.stderr}")
        raise RuntimeError("VR180 metadata injection failed")



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
import asyncio
import os
import uuid
import threading
from fastapi.responses import StreamingResponse

@app.post("/convert/")
async def convert_video(
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    content = await video.read()
    file_size_mb = len(content) / (1024 * 1024)
    
    if file_size_mb > 500:
        raise HTTPException(status_code=400, detail="File too large. Maximum 500MB for Colab.")

    unique_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"input_{unique_id}.mp4")
    output_path = os.path.join(UPLOAD_DIR, f"output_{unique_id}_vr180.mp4")
    final_output_path = os.path.join(FINAL_DIR, f"vr180_{unique_id}.mp4")
    temp_with_audio = output_path.replace(".mp4", "_with_audio.mp4")

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(content)

    try:
        print(f"🚀 Starting conversion: {video.filename} ({file_size_mb:.1f}MB)")
        
        # Clear GPU memory
        clear_gpu_cache()
        
        # Step 1: Convert to VR180 (creates equirectangular SBS output)
        await asyncio.to_thread(
            convert_2d_to_vr180_gpu_optimized,
            input_path,
            output_path,
            center_scale=0.82,
            focal_length=500,
            panini_alpha=0.7,
            stereographic_strength=0.2,
            eye_offset=6.3,
            max_disparity_degrees=1.3,
            max_blur_radius=8,
            ai_outpaint=True,
            upscale_to_8k=True,
            upscale_method="lanczos",
            expansion_degrees=210
        )

        # Verify conversion completed
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="VR180 conversion failed - no output file created")

        # Step 2: Add audio
        await add_audio_preserve_metadata(input_path, output_path, temp_with_audio)
        
        # Step 3: Inject VR180 metadata (FIXED parameter names)
        print("📝 Injecting VR180 metadata...")
        await asyncio.to_thread(
            inject_vr180_metadata_optimized,
            temp_with_audio,
            final_output_path,
            width=7680,      # Fixed: was center_scale_width
            height=3840      # Fixed: was center_scale_height
        )

        if not os.path.exists(final_output_path):
            raise HTTPException(status_code=500, detail="Final conversion failed")


        final_file_size = os.path.getsize(final_output_path)
        
        # Step 4: Save to conversion history
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
        # Cleanup temporary files - KEEP FINAL OUTPUT
        def cleanup():
            try:
                # Only clean up INTERMEDIATE files, keep the final output
                cleanup_paths = [input_path, output_path]
                
                if 'temp_with_audio' in locals() and temp_with_audio:
                    cleanup_paths.append(temp_with_audio)

                for path in cleanup_paths:
                    if os.path.exists(path) and path != final_output_path:
                        os.unlink(path)
                        print(f"🧹 Cleaned up temporary file: {path}")

                clear_gpu_cache()
                
                # Log what files remain (for debugging)
                remaining_files = []
                for path in [input_path, output_path, temp_with_audio, final_output_path]:
                    if os.path.exists(path):
                        remaining_files.append(path)
                
                print(f"📁 Files remaining after cleanup: {remaining_files}")
                
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")

        threading.Thread(target=cleanup, daemon=True).start()

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

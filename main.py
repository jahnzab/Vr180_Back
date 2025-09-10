# main.py
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
        created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))  # Add this line
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
    allow_origins=["http://localhost:3000", "https://vr180-frontend-new.vercel.app"],
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

# import sys
# import requests

# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import StreamingResponse
# import httpx



# ALO3D_BEARER_KEY = "jjPP43dkEJHzzzlIYSGVt8HE5xOAtz1"
# ALO3D_AUTH_URL = "https://auth.alo.ai/v1/auth"
# ALO3D_USER_ID = "A744d43X-3316-56f7-a6a3-99ecbbd44agg"
# ALO3D_INCIDENT_URL = "https://api.alo.ai/v1/vms/incidents"

# # -------------------------------
# # Inject VR180 Metadata
# # -------------------------------
# def inject_vr180_metadata(input_video_path: str, output_video_path: str):
#     """Inject VR180 metadata using Google's spatialmedia injector."""
#     try:
#         cmd = [
#             sys.executable, "-m", "spatialmedia",
#             "-i", "--stereo", "left-right", "--projection", "equirectangular",
#             input_video_path, output_video_path
#         ]
#         print(f"Running metadata injection: {' '.join(cmd)}")
#         subprocess.run(cmd, check=True, capture_output=True, text=True)
#         print(f"Metadata injected successfully into {output_video_path}")
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"Metadata injection failed: {e.stderr if e.stderr else e}")

# # -------------------------------
# # Get Temporary ALO3D Token
# # -------------------------------
# async def get_alo3d_token() -> str:
#     payload = {
#         "user": {"id": ALO3D_USER_ID},
#         "ttl": 60
#     }
#     headers = {
#         "Authorization": f"Bearer {ALO3D_BEARER_KEY}",
#         "accept": "application/json",
#         "content-type": "application/json"
#     }
#     async with httpx.AsyncClient(timeout=30) as client:
#         resp = await client.post(ALO3D_AUTH_URL, json=payload, headers=headers)
#         if resp.status_code == 200:
#             token = resp.json().get("token")
#             if not token:
#                 raise HTTPException(status_code=500, detail="ALO3D token not found")
#             return token
#         else:
#             raise HTTPException(status_code=500, detail=f"ALO3D auth failed: {resp.text}")

# # -------------------------------
# # Multipart Upload Helpers
# # -------------------------------
# async def create_multipart_upload(incident_id: str, file_id: str, filename: str, token: str):
#     url = f"{ALO3D_INCIDENT_URL}/{incident_id}/create_multipart_upload"
#     params = {"file_id": file_id, "filename": filename}
#     headers = {"x-api-key": token, "Content-Type": "application/json"}
#     async with httpx.AsyncClient(timeout=30) as client:
#         resp = await client.get(url, headers=headers, params=params)
#         resp.raise_for_status()
#         return resp.json()

# async def get_upload_urls(incident_id: str, upload_id: str, parts: list, token: str):
#     url = f"{ALO3D_INCIDENT_URL}/{incident_id}/upload_urls"
#     headers = {"x-api-key": token, "Content-Type": "application/json"}
#     data = {"upload_id": upload_id, "parts": parts}
#     async with httpx.AsyncClient(timeout=60) as client:
#         resp = await client.post(url, json=data, headers=headers)
#         resp.raise_for_status()
#         return resp.json()

# async def complete_multipart_upload(incident_id: str, upload_id: str, parts: list, token: str):
#     url = f"{ALO3D_INCIDENT_URL}/{incident_id}/complete_multipart_upload"
#     headers = {"x-api-key": token, "Content-Type": "application/json"}
#     data = {"upload_id": upload_id, "parts": parts}
#     async with httpx.AsyncClient(timeout=30) as client:
#         resp = await client.post(url, json=data, headers=headers)
#         resp.raise_for_status()
#         return resp.json()

# # -------------------------------
# # Convert Video Endpoint
# # -------------------------------
# @app.post("/convert/")
# async def convert_video(video: UploadFile = File(...), incident_id: str = "default_incident"):
#     if not video.content_type.startswith("video/"):
#         raise HTTPException(status_code=400, detail="File must be a video")

#     # Save temp file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
#         temp_file.write(await video.read())
#         temp_file_path = temp_file.name
#     print(f"Temporary file created: {temp_file_path}")

#     output_path = temp_file_path.replace(".mp4", "_vr180.mp4")

#     try:
#         # Step 1: Get temporary token
#         token = await get_alo3d_token()

#         # Step 2: Create multipart upload
#         file_id = os.path.basename(temp_file_path)
#         multipart_info = await create_multipart_upload(incident_id, file_id, video.filename, token)
#         upload_id = multipart_info["upload_id"]
#         parts = multipart_info["parts"]

#         # Step 3: Get upload URLs
#         upload_urls_info = await get_upload_urls(incident_id, upload_id, parts, token)
#         upload_urls = upload_urls_info["upload_urls"]

#         # Step 4: Upload chunks directly to S3
#         for idx, part_url in enumerate(upload_urls):
#             with open(temp_file_path, "rb") as f:
#                 chunk = f.read()  # For simplicity, read whole file. For large files, read per part size
#                 async with httpx.AsyncClient(timeout=300) as client:
#                     resp = await client.put(part_url, content=chunk)
#                     resp.raise_for_status()
#             parts[idx]["etag"] = resp.headers.get("ETag", "")

#         # Step 5: Complete multipart upload
#         await complete_multipart_upload(incident_id, upload_id, parts, token)

#         # Step 6: VR180 Conversion & Metadata Injection
#         await asyncio.to_thread(inject_vr180_metadata, temp_file_path, output_path)

#     except Exception as e:
#         print(f"Error during multipart upload & conversion: {e}")
#         # Fallback: inject metadata only
#         await asyncio.to_thread(inject_vr180_metadata, temp_file_path, output_path)

#     # Stream back converted file
#     def iter_file(path: str):
#         with open(path, "rb") as f:
#             while chunk := f.read(8192):
#                 yield chunk

#     headers = {
#         "Content-Disposition": f'attachment; filename="vr180_{video.filename}"',
#         "Content-Type": "video/mp4"
#     }

#     # Cleanup temp files in background
#     def cleanup():
#         try:
#             if os.path.exists(temp_file_path):
#                 os.unlink(temp_file_path)
#             if os.path.exists(output_path):
#                 import time; time.sleep(30)
#                 os.unlink(output_path)
#         except Exception as e:
#             print(f"Cleanup error: {e}")

#     threading.Thread(target=cleanup, daemon=True).start()

#     return StreamingResponse(iter_file(output_path), headers=headers, media_type="video/mp4")

import os
import tempfile
import asyncio
import threading
import subprocess
import requests
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse



# -------------------------------
# Immersity.ai Config
# -------------------------------
# CLIENT_ID = "<YOUR_CLIENT_ID>"
# CLIENT_SECRET = "<YOUR_CLIENT_SECRET>"
# TOKEN_URL = "https://auth.immersity.ai/auth/realms/immersity/protocol/openid-connect/token"
# CONVERT_URL = "https://app.immersity.ai/upload"


# # -------------------------------
# # Get Access Token
# # -------------------------------
# async def get_immersity_token() -> str:
#     data = {
#         "grant_type": "client_credentials",
#         "client_id": CLIENT_ID,
#         "client_secret": CLIENT_SECRET,
#     }
#     headers = {"Content-Type": "application/x-www-form-urlencoded"}

#     async with httpx.AsyncClient(timeout=30) as client:
#         resp = await client.post(TOKEN_URL, data=data, headers=headers)
#         if resp.status_code == 200:
#             token = resp.json().get("access_token")
#             if not token:
#                 raise HTTPException(status_code=500, detail="Immersity token not found")
#             return token
#         else:
#             raise HTTPException(status_code=500, detail=f"Auth failed: {resp.text}")


# # -------------------------------
# # Inject VR180 Metadata
# # -------------------------------
# def inject_vr180_metadata(input_video_path: str, output_video_path: str):
#     try:
#         cmd = [
#             "python3", "-m", "spatialmedia",
#             "-i", "--stereo", "left-right", "--projection", "equirectangular",
#             input_video_path, output_video_path
#         ]
#         subprocess.run(cmd, check=True, capture_output=True, text=True)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(f"Metadata injection failed: {e.stderr if e.stderr else e}")


# # -------------------------------
# # Convert Video Endpoint
# # -------------------------------
# @app.post("/convert/")
# async def convert_video(video: UploadFile = File(...)):
#     if not video.content_type.startswith("video/"):
#         raise HTTPException(status_code=400, detail="File must be a video")

#     # Save video to temp
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
#         temp_file.write(await video.read())
#         temp_file_path = temp_file.name

#     try:
#         # Step 1: Get OAuth2 token
#         token = await get_immersity_token()
#         headers = {"Authorization": f"Bearer {token}"}

#         # Step 2: Upload to Immersity
#         with open(temp_file_path, "rb") as f:
#             files = {"file": (video.filename, f, video.content_type)}
#             resp = requests.post(CONVERT_URL, headers=headers, files=files)

#         if resp.status_code != 200:
#             raise HTTPException(status_code=500, detail=f"Immersity upload failed: {resp.text}")

#         # Step 3: Parse response
#         # Immersity usually gives JSON with video id / playback URL
#         if "application/json" in resp.headers.get("Content-Type", ""):
#             result = resp.json()
#             return JSONResponse(content={"status": "success", "immersity_response": result})
#         else:
#             # If Immersity actually returns raw video (less likely)
#             output_path = temp_file_path.replace(".mp4", "_vr180.mp4")
#             with open(output_path, "wb") as out:
#                 out.write(resp.content)

#             await asyncio.to_thread(inject_vr180_metadata, output_path, output_path)

#             def iter_file(path: str):
#                 with open(path, "rb") as f:
#                     while chunk := f.read(8192):
#                         yield chunk

#             headers = {
#                 "Content-Disposition": f'attachment; filename="vr180_{video.filename}"',
#                 "Content-Type": "video/mp4"
#             }
#             return StreamingResponse(iter_file(output_path), headers=headers, media_type="video/mp4")

#     finally:
#         # Cleanup
#         def cleanup():
#             try:
#                 if os.path.exists(temp_file_path):
#                     os.unlink(temp_file_path)
#             except Exception as e:
#                 print(f"Cleanup error: {e}")

#         threading.Thread(target=cleanup, daemon=True).start()


import os
import uuid
import asyncio
import threading
from datetime import datetime
from urllib.parse import unquote
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
import os
import sys
import tempfile
import subprocess
import threading
import asyncio
import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import shutil
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from datetime import datetime, timezone
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREVIEW_DIR = "temp_videos"
os.makedirs(PREVIEW_DIR, exist_ok=True)

# -------------------------------
# Convert Endpoint
# -------------------------------
from midas.model import MidasNet_small
from midas.transforms import small_transform
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model directly from the .pt file
midas = torch.load("midas_v21_small_256.pt", map_location=device)  # adjust path if needed
midas.to(device)
midas.eval()

# If you need transforms, you have to define them manually or copy from MiDaS repo
midas_transforms = None  # replace with actual preprocessing if required

print(midas)
print(midas_transforms)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")  # lightweight
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
# midas.to(device)
# midas.eval()
# @app.get("/working")
# async def working():
#     return {"message": "✅ System working - processing in background"}

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
# Depth estimation + smoothing
# -------------------------------
def estimate_depth(img):
    """
    Run MiDaS depth estimation on a single frame.
    """
    # Convert BGR to RGB if using OpenCV
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform -> [C, H, W]
    input_tensor = midas_transforms(img_rgb).to(device)

    # Add batch dimension -> [1, C, H, W]
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():
        prediction = midas(input_tensor)  # output [1, H_out, W_out] or [1, 1, H_out, W_out]

    # Ensure 4D for interpolate
    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(1)  # [1, 1, H, W]

    # Resize to original frame size
    prediction_resized = torch.nn.functional.interpolate(
        prediction, size=img.shape[:2], mode="bicubic", align_corners=False
    ).squeeze(0).squeeze(0)  # remove batch & channel dims

    # Normalize depth
    depth = prediction_resized.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth


def temporal_smooth(prev_depth, curr_depth, alpha=0.7):
    if prev_depth is None:
        return curr_depth
    return alpha * prev_depth + (1 - alpha) * curr_depth

def make_stereo_pair(img, depth, eye_offset=6):
    """Create left/right stereo views from depth map."""
    h, w = img.shape[:2]
    depth_resized = cv2.resize(depth, (w, h))  # match depth to frame
    left = np.zeros_like(img)
    right = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            shift = int((1 - depth_resized[y, x]) * eye_offset)
            lx = min(w - 1, max(0, x - shift))
            rx = min(w - 1, max(0, x + shift))
            left[y, lx] = img[y, x]
            right[y, rx] = img[y, x]
    return left, right


# method for injector vr 180


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
def convert_2d_to_vr180(input_path: str, output_path: str):
    temp_dir = tempfile.mkdtemp(dir=BASE_DIR)
    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_paths = []
    prev_depth = None

    # Frame-by-frame conversion
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        depth = estimate_depth(frame)
        depth = temporal_smooth(prev_depth, depth)
        prev_depth = depth

        left, right = make_stereo_pair(frame, depth)
        sbs = np.concatenate((left, right), axis=1)

        frame_path = os.path.join(temp_dir, f"{len(frame_paths):05d}.png")
        cv2.imwrite(frame_path, sbs)
        frame_paths.append(frame_path)

    cap.release()

    if not frame_paths:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError("No frames were processed. Check input video.")

    # Write SBS video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    for f in frame_paths:
        img = cv2.imread(f)
        if img is None:
            continue
        out.write(img)
        os.unlink(f)

    out.release()
    shutil.rmtree(temp_dir, ignore_errors=True)

    if not os.path.exists(output_path):
        raise RuntimeError("Failed to create SBS output video.")

    # Inject VR180 metadata safely
    temp_output = output_path + ".tmp.mp4"
    shutil.move(output_path, temp_output)

    try:
        inject_vr180_metadata(temp_output, output_path)
    except Exception as e:
        shutil.move(temp_output, output_path)  # fallback to SBS only
        raise RuntimeError(f"Metadata injection failed: {e}")
    finally:
        if os.path.exists(temp_output):
            os.remove(temp_output)

#  work good with audio but without preveiw

from moviepy.editor import VideoFileClip
import asyncio
import subprocess
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "tmp_uploads")
FINAL_DIR = os.path.join(BASE_DIR, "videos")   # ✅ permanent storage
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# Mock DB for history
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

@app.post("/convert/")
async def convert_video(
    video: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    unique_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"input_{unique_id}.mp4")
    output_path = os.path.join(UPLOAD_DIR, f"output_{unique_id}_vr180.mp4")
    final_output_path = os.path.join(FINAL_DIR, f"vr180_{unique_id}.mp4")

    with open(input_path, "wb") as f:
        f.write(await video.read())

    try:
        # Step 1: Convert to VR180
        await asyncio.to_thread(convert_2d_to_vr180, input_path, output_path)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="VR180 conversion failed.")

        # Step 2: Add audio with metadata
        await add_audio_preserve_metadata(input_path, output_path, final_output_path)

        if not os.path.exists(final_output_path):
            raise HTTPException(status_code=500, detail="Final conversion failed.")

        # Step 3: Save to history with database
        file_size = os.path.getsize(final_output_path) if os.path.exists(final_output_path) else None
        
        conversion = save_to_history(
            db=db,
            user_id=current_user.id,
            original_filename=video.filename,
            output_path=final_output_path,
            file_size=file_size
        )
        
        # Step 4: Return streaming response (for download)
        def iter_file(path: str):
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk

        return StreamingResponse(
            iter_file(final_output_path),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f'attachment; filename="vr180_{video.filename}"',
                "X-Conversion-ID": str(conversion.id)
            }
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error and re-raise
        print(f"Conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # ✅ Only clean up temporary intermediate files, not the final output
        def cleanup():
            try:
                # Only delete the intermediate files, not the final output
                for path in [input_path, output_path]:
                    if os.path.exists(path):
                        os.unlink(path)
                        print(f"Cleaned up temporary file: {path}")
            except Exception as e:
                print(f"Cleanup error: {e}")

        threading.Thread(target=cleanup, daemon=True).start()

@app.get("/user/conversions/")
async def get_user_conversions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Get conversions from database for the current user
    conversions = db.query(Conversion).filter(
        Conversion.user_id == current_user.id
    ).order_by(Conversion.created_at.desc()).all()
    
    # Convert to list of dictionaries for JSON response
    return [
        {
            "id": conv.id,
            "original_filename": conv.original_filename,
            "created_at": conv.created_at,
            "file_size": conv.file_size
        }
        for conv in conversions
    ]
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
from fastapi.responses import FileResponse
from fastapi import HTTPException, Depends
from urllib.parse import unquote
import os
from sqlalchemy.orm import Session

# Make sure this function is defined
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
@app.delete("/conversions/{conversion_id}")
async def delete_conversion(
    conversion_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Find the conversion
        conversion = get_conversion_by_id(db, conversion_id, current_user.id)
        
        if not conversion:
            raise HTTPException(status_code=404, detail="Conversion not found")
        
        # Delete the file from storage
        if conversion.output_path and os.path.exists(conversion.output_path):
            os.remove(conversion.output_path)
        
        # Delete the record from database
        db.delete(conversion)
        db.commit()
        
        return {"message": "Conversion deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
async def add_audio_preserve_metadata(original_path, vr180_video_path, output_path):
    """Add audio to VR180 video while preserving metadata using FFmpeg"""
    cmd = [
        'ffmpeg',
        '-i', vr180_video_path,  # VR180 video with metadata
        '-i', original_path,     # Original video with audio
        '-map', '0:v',          # Use video from first input (VR180)
        '-map', '1:a',          # Use audio from second input (original)
        '-c:v', 'copy',         # Copy video stream (preserves metadata!)
        '-c:a', 'aac',          # Encode audio to AAC
        '-movflags', 'use_metadata_tags',  # Preserve metadata
        '-map_metadata', '0',   # Copy metadata from first input
        '-y',                   # Overwrite output
        output_path
    ]
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode()
            print(f"FFmpeg error: {error_msg}")
            raise Exception(f"Audio addition failed: {error_msg}")
            
    except Exception as e:
        print(f"Audio addition error: {e}")
        # Fallback: if audio addition fails, use the VR180 video without audio
        os.replace(vr180_video_path, output_path)


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
@app.post("/feedback/")
async def create_feedback(
    feedback: FeedbackCreate, 
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_feedback = Feedback(
        user_id=current_user.id,
        content=feedback.content
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    
    return {"message": "Feedback submitted successfully"}


@app.get("/user/conversions/")
async def get_user_conversions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    conversions = db.query(ConversionHistory).filter(
        ConversionHistory.user_id == current_user.id
    ).order_by(ConversionHistory.created_at.desc()).all()
    
    return [
        {
            "id": conv.id,
            "original_filename": conv.original_filename,
            "converted_filename": conv.converted_filename,
            "created_at": conv.created_at
        }
        for conv in conversions
    ]
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
import uvicorn
port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=port)

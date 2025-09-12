def inject_vr180_metadata(
    input_video_path: str,
    output_video_path: str,
    center_scale_width: int = 7680,    # 8K width (4x upscaled from 1920)
    center_scale_height: int = 4320,   # 8K height (4x upscaled from 1080)
    cropped_image_width: int = 7680,   # Final output width
    cropped_image_height: int = 4320,  # Final output height (full resolution)
    cropped_left: int = 0,
    cropped_top: int = 0                # 0 for side-by-side (no vertical crop)
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

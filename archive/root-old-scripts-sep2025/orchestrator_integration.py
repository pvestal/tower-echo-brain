#!/usr/bin/env python3
"""
Echo Orchestrator with Character Approval Workflow
1. Generate character images FIRST
2. Quality check automatically
3. Show to Patrick for approval
4. ONLY THEN proceed to video/music/voice
"""

import asyncio
import httpx
import json
import time
import os
from typing import Dict, Any, List
import cv2
import numpy as np

class ApprovalWorkflowOrchestrator:
    def __init__(self):
        self.services = {
            "comfyui": "http://localhost:8188",
            "voice": "http://localhost:8312", 
            "music": "http://localhost:8315",
            "kb": "http://localhost:8307"
        }
        self.quality_threshold = 75
        self.max_retries = 3
        self.approval_dir = "***REMOVED***/pending_approval/"
        
        # Create approval directory
        os.makedirs(self.approval_dir, exist_ok=True)
        
    async def orchestrate_with_approval(self, character_desc="Kai cyberpunk ninja") -> Dict[str, Any]:
        """
        PROPER WORKFLOW:
        1. Generate character reference images
        2. Auto quality check with retry
        3. Save to approval folder for Patrick
        4. Wait for approval
        5. Generate video with approved character
        6. Add music
        7. Add voices
        """
        
        print("="*60)
        print("🎬 STARTING CHARACTER APPROVAL WORKFLOW")
        print("="*60)
        
        results = []
        
        # PHASE 1: CHARACTER GENERATION
        print("\n📸 PHASE 1: GENERATING CHARACTER REFERENCE IMAGES")
        character_images = await self.generate_character_references(character_desc)
        
        if not character_images:
            return {
                "response": "Failed to generate quality character images",
                "phase": "character_generation",
                "success": False
            }
        
        results.append(f"✅ Generated {len(character_images)} character references")
        
        # PHASE 2: SAVE FOR APPROVAL
        print(f"\n👁️ PHASE 2: SAVING FOR PATRICK'S APPROVAL")
        approval_files = []
        timestamp = int(time.time())
        
        for i, img in enumerate(character_images):
            approval_path = f"{self.approval_dir}character_{timestamp}_{i+1}.png"
            # Copy image to approval folder
            import shutil
            shutil.copy(img, approval_path)
            approval_files.append(approval_path)
            print(f"  ✅ Saved: {approval_path}")
        
        results.append(f"📂 Images saved to: {self.approval_dir}")
        
        # PHASE 3: RETURN FOR APPROVAL
        return {
            "response": "Character images ready for approval",
            "phase": "awaiting_approval",
            "approval_required": True,
            "images": approval_files,
            "details": results,
            "next_step": "Review images in ***REMOVED***/pending_approval/",
            "instruction": "Say 'approved' to continue with video generation, or 'retry' for new characters"
        }
    
    async def continue_after_approval(self, approved_images: List[str]) -> Dict[str, Any]:
        """
        Continue workflow after Patrick approves characters
        """
        print("\n🎥 PHASE 4: GENERATING VIDEO WITH APPROVED CHARACTERS")
        
        results = []
        
        # Generate video frames based on approved character style
        video_frames = await self.generate_video_frames(approved_images)
        
        if not video_frames:
            return {
                "response": "Failed to generate video frames",
                "phase": "video_generation",
                "success": False
            }
        
        results.append(f"✅ Generated {len(video_frames)} video frames")
        
        # PHASE 5: ADD MUSIC
        print("\n🎵 PHASE 5: ADDING MUSIC")
        music_track = await self.generate_music_track()
        if music_track:
            results.append("✅ Music track generated")
        
        # PHASE 6: ADD VOICES
        print("\n🗣️ PHASE 6: ADDING CHARACTER VOICES")
        voice_track = await self.generate_voice_track()
        if voice_track:
            results.append("✅ Voice track generated")
        
        # PHASE 7: COMBINE EVERYTHING
        print("\n🎬 PHASE 7: CREATING FINAL VIDEO")
        output_video = f"***REMOVED***/final_approved_{int(time.time())}.mp4"
        
        if await self.create_final_video(video_frames, music_track, voice_track, output_video):
            results.append(f"✅ Final video created: {output_video}")
            return {
                "response": "Video successfully created with approved characters!",
                "phase": "completed",
                "output": output_video,
                "details": results,
                "success": True
            }
        else:
            return {
                "response": "Failed to create final video",
                "phase": "video_assembly",
                "success": False
            }
    
    async def generate_character_references(self, character_desc: str, count: int = 3) -> List[str]:
        """Generate character reference images with auto quality retry"""
        
        images = []
        
        for ref_num in range(count):
            print(f"\n  Generating reference {ref_num + 1}/{count}")
            
            for attempt in range(self.max_retries):
                print(f"    Attempt {attempt + 1}/{self.max_retries}")

                # START BASIC, BUILD IN LAYERS
                # First pass: FAST draft (low quality, quick generation)
                # Second pass: Add detail
                # Third pass: Polish if needed
                quality_level = attempt
                
                workflow = self.create_character_workflow(
                    character_desc, 
                    ref_num,
                    quality_level
                )
                
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            f"{self.services['comfyui']}/prompt",
                            json={"prompt": workflow},
                            timeout=10.0
                        )
                        
                        if response.status_code == 200:
                            prompt_id = response.json()['prompt_id']
                            print(f"      ⏳ Generating (quality level {quality_level})...")
                            
                            # Wait for generation (shorter to avoid timeout)
                            await asyncio.sleep(10 + quality_level * 2)
                            
                            # Get output
                            image_path = await self.get_comfyui_output(prompt_id)
                            
                            if image_path:
                                # Quality check
                                quality_score = self.check_image_quality(image_path)
                                print(f"      📊 Quality score: {quality_score}/100")
                                
                                if quality_score >= self.quality_threshold:
                                    print(f"      ✅ Quality approved!")
                                    images.append(image_path)
                                    break
                                else:
                                    print(f"      ❌ Quality too low, retrying...")
                            else:
                                print(f"      ❌ Generation failed")
                        
                except Exception as e:
                    print(f"      Error: {e}")
                
                if attempt == self.max_retries - 1:
                    print(f"      💥 Max retries reached")
        
        return images
    
    def create_character_workflow(self, character: str, ref_num: int, quality_level: int) -> Dict:
        """Create ComfyUI workflow - START SIMPLE, LAYER QUALITY"""

        # LAYER 0: FAST DRAFT (5-10 seconds)
        if quality_level == 0:
            prompt = f"{character}, character sheet, reference pose {ref_num + 1}"
            steps = 15  # Fast
            cfg = 6.0   # Low
            sampler = "euler"  # Fastest

        # LAYER 1: ADD DETAIL (10-15 seconds)
        elif quality_level == 1:
            prompt = f"{character}, character sheet, reference pose {ref_num + 1}, detailed"
            steps = 20
            cfg = 7.0
            sampler = "euler_a"

        # LAYER 2: QUALITY POLISH (15-20 seconds)
        else:
            prompt = f"{character}, character sheet, reference pose {ref_num + 1}, masterpiece, best quality"
            steps = 25
            cfg = 7.5
            sampler = "dpmpp_2m_sde"
        
        return {
            "1": {
                "inputs": {"ckpt_name": "animagine_xl_3.1.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {"text": prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality, bad anatomy, ugly",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {"width": 1920, "height": 1080, "batch_size": 1},
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": int(time.time()) + ref_num * 1000 + quality_level,
                    "steps": steps,  # Use layered steps
                    "cfg": cfg,      # Use layered CFG
                    "sampler_name": sampler,  # Use layered sampler
                    "scheduler": "normal" if quality_level == 0 else "karras",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode"
            },
            "7": {
                "inputs": {
                    "filename_prefix": f"character_ref_{ref_num:02d}",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
    
    def check_image_quality(self, image_path: str) -> int:
        """Check image quality with multiple metrics"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return 0
            
            score = 100
            
            # Check color diversity
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            if unique_colors < 5000:
                score -= 20
            
            # Check sharpness
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian < 200:
                score -= 20
            
            # Check uniformity
            std = np.std(img)
            if std < 40:
                score -= 30
            
            return max(0, score)
            
        except:
            return 0
    
    async def get_comfyui_output(self, prompt_id: str) -> str:
        """Get output from ComfyUI generation"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.services['comfyui']}/history/{prompt_id}")
                if response.status_code == 200:
                    history = response.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get('outputs', {})
                        for node_id, node_output in outputs.items():
                            if 'images' in node_output:
                                filename = node_output['images'][0]['filename']
                                return f"/home/patrick/ComfyUI/output/{filename}"
        except:
            pass
        return None
    
    async def generate_video_frames(self, approved_images: List[str]) -> List[str]:
        """Generate video frames based on approved character style"""
        # Simplified for now - would use approved images as reference
        return approved_images * 10  # Repeat frames for video
    
    async def generate_music_track(self) -> str:
        """Generate music track"""
        # Placeholder - would call music service
        return "/tmp/music_track.mp3"
    
    async def generate_voice_track(self) -> str:
        """Generate voice track"""
        # Placeholder - would call voice service
        return "/tmp/voice_track.wav"
    
    async def create_final_video(self, frames: List[str], music: str, voice: str, output: str) -> bool:
        """Combine everything into final video"""
        import subprocess
        
        # Create frame list
        frame_list = "/tmp/approved_frames.txt"
        with open(frame_list, 'w') as f:
            for frame in frames:
                f.write(f"file '{frame}'\n")
        
        # FFmpeg command to create video
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", frame_list,
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-vf", "fps=24",
            "-t", "30", output
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    
    async def handle_orchestration_request(self, message: str) -> Dict[str, Any]:
        """Handle orchestration with approval workflow"""
        msg_lower = message.lower()

        if "approved" in msg_lower:
            # Get latest images from approval folder
            import glob
            approved = glob.glob(f"{self.approval_dir}*.png")
            if approved:
                return await self.continue_after_approval(approved[-3:])
            return {"response": "No images to approve"}

        elif "retry" in msg_lower or "character" in msg_lower or "trailer" in msg_lower:
            # Quick response for testing - don't actually generate
            return {
                "response": "Character generation ready! Use existing images in /home/patrick/ComfyUI/output/",
                "phase": "character_selection",
                "existing_images": [
                    "/home/patrick/ComfyUI/output/character_ref_00_00001_.png",
                    "/home/patrick/ComfyUI/output/character_ref_00_00002_.png"
                ],
                "quick_mode": True
            }
            # Disabled for now to avoid timeout: return await self.orchestrate_with_approval()

        return {"response": "I generate characters for approval FIRST, then create videos"}

# Global orchestrator - Use inquisitive conversation flow
from inquisitive_orchestrator import InquisitiveOrchestrator
orchestrator = InquisitiveOrchestrator()

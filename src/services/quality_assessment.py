#!/usr/bin/env python3
"""
Advanced Video Quality Assessment System for Echo Brain
Analyzes generated videos for technical quality, artistic merit, and prompt adherence
"""

import os
import json
import logging
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageStat
import torch
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
import requests
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoQualityAssessment:
    def __init__(self):
        """Initialize the quality assessment system"""
        self.temp_dir = "/tmp/quality_assessment"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Initialize CLIP model for prompt adherence
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_available = True
            logger.info("✅ CLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ CLIP model not available: {e}")
            self.clip_available = False

        # Quality thresholds
        self.thresholds = {
            "minimum_score": 70,
            "excellent_score": 90,
            "frame_consistency_threshold": 0.85,
            "minimum_resolution": (512, 512),
            "target_resolution": (1024, 1024),
            "noise_threshold": 0.15,
            "blur_threshold": 100,
            "prompt_similarity_threshold": 0.7
        }

        # Create quality database
        self.quality_db_path = "/home/patrick/Documents/Tower/core-services/echo-brain/quality_scores.json"
        self.load_quality_database()

    def load_quality_database(self):
        """Load existing quality scores database"""
        try:
            if os.path.exists(self.quality_db_path):
                with open(self.quality_db_path, 'r') as f:
                    self.quality_db = json.load(f)
            else:
                self.quality_db = {"assessments": [], "statistics": {}}
            logger.info(f"📊 Quality database loaded: {len(self.quality_db.get('assessments', []))} records")
        except Exception as e:
            logger.error(f"❌ Failed to load quality database: {e}")
            self.quality_db = {"assessments": [], "statistics": {}}

    def save_quality_database(self):
        """Save quality scores to database"""
        try:
            with open(self.quality_db_path, 'w') as f:
                json.dump(self.quality_db, f, indent=2)
            logger.info("💾 Quality database saved")
        except Exception as e:
            logger.error(f"❌ Failed to save quality database: {e}")

    def assess_video_quality(self, video_path: str, prompt: str = "", metadata: Dict = None) -> Dict[str, Any]:
        """
        Comprehensive video quality assessment

        Args:
            video_path: Path to the video file
            prompt: Original prompt used for generation
            metadata: Additional metadata about the video

        Returns:
            Detailed quality assessment report
        """
        logger.info(f"🔍 Starting quality assessment for: {video_path}")

        if not os.path.exists(video_path):
            return {"error": "Video file not found", "overall_score": 0}

        assessment = {
            "video_path": video_path,
            "video_hash": self._get_file_hash(video_path),
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "metadata": metadata or {},
            "technical_quality": {},
            "artistic_quality": {},
            "prompt_adherence": {},
            "overall_score": 0,
            "recommendations": [],
            "pass_threshold": False
        }

        try:
            # Extract video information
            video_info = self._get_video_info(video_path)
            assessment["video_info"] = video_info

            # Extract frames for analysis
            frames = self._extract_frames(video_path)
            if not frames:
                assessment["error"] = "Failed to extract frames"
                return assessment

            # Technical Quality Assessment
            assessment["technical_quality"] = self._assess_technical_quality(frames, video_info)

            # Artistic Quality Assessment
            assessment["artistic_quality"] = self._assess_artistic_quality(frames)

            # Prompt Adherence Assessment (if CLIP is available and prompt provided)
            if self.clip_available and prompt:
                assessment["prompt_adherence"] = self._assess_prompt_adherence(frames, prompt)

            # Calculate overall score
            assessment["overall_score"] = self._calculate_overall_score(assessment)

            # Generate recommendations
            assessment["recommendations"] = self._generate_recommendations(assessment)

            # Check if video passes quality threshold
            assessment["pass_threshold"] = assessment["overall_score"] >= self.thresholds["minimum_score"]

            # Save to database
            self._save_assessment(assessment)

            logger.info(f"✅ Quality assessment completed. Score: {assessment['overall_score']:.1f}/100")

        except Exception as e:
            logger.error(f"❌ Quality assessment failed: {e}")
            assessment["error"] = str(e)

        return assessment

    def _get_file_hash(self, file_path: str) -> str:
        """Generate SHA-256 hash of file for tracking"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]  # First 16 chars for brevity

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                info = json.loads(result.stdout)

                # Extract video stream info
                video_stream = None
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                        break

                if video_stream:
                    return {
                        "width": int(video_stream.get('width', 0)),
                        "height": int(video_stream.get('height', 0)),
                        "duration": float(video_stream.get('duration', 0)),
                        "fps": eval(video_stream.get('r_frame_rate', '0/1')),
                        "codec": video_stream.get('codec_name', 'unknown'),
                        "bitrate": int(video_stream.get('bit_rate', 0)),
                        "file_size": int(info.get('format', {}).get('size', 0))
                    }

            return {"error": "Could not extract video info"}
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            return {"error": str(e)}

    def _extract_frames(self, video_path: str, max_frames: int = 10) -> List[np.ndarray]:
        """Extract frames from video for analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR_RGB))

            cap.release()
            return frames
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []

    def _assess_technical_quality(self, frames: List[np.ndarray], video_info: Dict) -> Dict[str, Any]:
        """Assess technical aspects of video quality"""
        scores = {}

        try:
            # Resolution Quality
            width = video_info.get('width', 0)
            height = video_info.get('height', 0)
            min_res = self.thresholds["minimum_resolution"]
            target_res = self.thresholds["target_resolution"]

            if width >= target_res[0] and height >= target_res[1]:
                resolution_score = 100
            elif width >= min_res[0] and height >= min_res[1]:
                resolution_score = 70 + (min(width/target_res[0], height/target_res[1]) * 30)
            else:
                resolution_score = 30

            scores["resolution"] = {
                "score": min(100, resolution_score),
                "actual": (width, height),
                "target": target_res,
                "minimum": min_res
            }

            # Frame Consistency
            if len(frames) > 1:
                consistency_scores = []
                for i in range(len(frames) - 1):
                    similarity = self._calculate_frame_similarity(frames[i], frames[i + 1])
                    consistency_scores.append(similarity)

                avg_consistency = np.mean(consistency_scores)
                consistency_score = min(100, avg_consistency * 100)

                scores["frame_consistency"] = {
                    "score": consistency_score,
                    "average_similarity": avg_consistency,
                    "frame_similarities": consistency_scores,
                    "threshold": self.thresholds["frame_consistency_threshold"]
                }
            else:
                scores["frame_consistency"] = {"score": 100, "note": "Single frame analysis"}

            # Sharpness/Blur Detection
            sharpness_scores = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness_scores.append(laplacian_var)

            avg_sharpness = np.mean(sharpness_scores)
            blur_threshold = self.thresholds["blur_threshold"]

            if avg_sharpness >= blur_threshold * 2:
                sharpness_score = 100
            elif avg_sharpness >= blur_threshold:
                sharpness_score = 50 + (avg_sharpness / blur_threshold) * 25
            else:
                sharpness_score = max(10, (avg_sharpness / blur_threshold) * 50)

            scores["sharpness"] = {
                "score": min(100, sharpness_score),
                "average_sharpness": avg_sharpness,
                "threshold": blur_threshold,
                "frame_sharpness": sharpness_scores
            }

            # Noise Detection
            noise_scores = []
            for frame in frames:
                noise_level = self._calculate_noise_level(frame)
                noise_scores.append(noise_level)

            avg_noise = np.mean(noise_scores)
            noise_threshold = self.thresholds["noise_threshold"]

            if avg_noise <= noise_threshold / 2:
                noise_score = 100
            elif avg_noise <= noise_threshold:
                noise_score = 50 + (1 - avg_noise / noise_threshold) * 50
            else:
                noise_score = max(10, 50 * (1 - min(avg_noise / noise_threshold, 2)))

            scores["noise"] = {
                "score": min(100, noise_score),
                "average_noise": avg_noise,
                "threshold": noise_threshold,
                "frame_noise": noise_scores
            }

            # Artifact Detection
            artifact_scores = []
            for frame in frames:
                artifacts = self._detect_artifacts(frame)
                artifact_scores.append(artifacts)

            avg_artifacts = np.mean(artifact_scores)
            artifact_score = max(10, 100 - (avg_artifacts * 100))

            scores["artifacts"] = {
                "score": artifact_score,
                "average_artifacts": avg_artifacts,
                "frame_artifacts": artifact_scores
            }

        except Exception as e:
            logger.error(f"Technical quality assessment error: {e}")
            scores["error"] = str(e)

        return scores

    def _assess_artistic_quality(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Assess artistic aspects of video quality"""
        scores = {}

        try:
            # Color Harmony
            color_scores = []
            for frame in frames:
                color_score = self._assess_color_harmony(frame)
                color_scores.append(color_score)

            scores["color_harmony"] = {
                "score": np.mean(color_scores),
                "frame_scores": color_scores
            }

            # Composition
            composition_scores = []
            for frame in frames:
                composition_score = self._assess_composition(frame)
                composition_scores.append(composition_score)

            scores["composition"] = {
                "score": np.mean(composition_scores),
                "frame_scores": composition_scores
            }

            # Visual Interest
            interest_scores = []
            for frame in frames:
                interest_score = self._assess_visual_interest(frame)
                interest_scores.append(interest_score)

            scores["visual_interest"] = {
                "score": np.mean(interest_scores),
                "frame_scores": interest_scores
            }

            # Dynamic Range
            dynamic_scores = []
            for frame in frames:
                dynamic_score = self._assess_dynamic_range(frame)
                dynamic_scores.append(dynamic_score)

            scores["dynamic_range"] = {
                "score": np.mean(dynamic_scores),
                "frame_scores": dynamic_scores
            }

        except Exception as e:
            logger.error(f"Artistic quality assessment error: {e}")
            scores["error"] = str(e)

        return scores

    def _assess_prompt_adherence(self, frames: List[np.ndarray], prompt: str) -> Dict[str, Any]:
        """Assess how well the video matches the original prompt using CLIP"""
        if not self.clip_available:
            return {"error": "CLIP model not available"}

        try:
            similarity_scores = []

            for frame in frames:
                # Convert numpy array to PIL Image
                image = Image.fromarray(frame)

                # Process with CLIP
                inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True)

                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    similarity = torch.nn.functional.softmax(logits_per_image, dim=1)[0, 0].item()
                    similarity_scores.append(similarity)

            avg_similarity = np.mean(similarity_scores)

            # Convert similarity to score (0-100)
            if avg_similarity >= self.thresholds["prompt_similarity_threshold"]:
                prompt_score = 70 + (avg_similarity - self.thresholds["prompt_similarity_threshold"]) * 100
            else:
                prompt_score = avg_similarity / self.thresholds["prompt_similarity_threshold"] * 70

            return {
                "score": min(100, prompt_score),
                "average_similarity": avg_similarity,
                "frame_similarities": similarity_scores,
                "threshold": self.thresholds["prompt_similarity_threshold"],
                "prompt": prompt
            }

        except Exception as e:
            logger.error(f"Prompt adherence assessment error: {e}")
            return {"error": str(e)}

    def _calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate similarity between two frames using structural similarity"""
        try:
            from skimage.metrics import structural_similarity as ssim

            # Convert to grayscale for SSIM
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            # Ensure same dimensions
            if gray1.shape != gray2.shape:
                min_h, min_w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
                gray1 = gray1[:min_h, :min_w]
                gray2 = gray2[:min_h, :min_w]

            similarity = ssim(gray1, gray2)
            return max(0, similarity)
        except ImportError:
            # Fallback to simple correlation
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY).flatten()
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY).flatten()
            correlation = np.corrcoef(gray1, gray2)[0, 1]
            return max(0, correlation) if not np.isnan(correlation) else 0

    def _calculate_noise_level(self, frame: np.ndarray) -> float:
        """Calculate noise level in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Use Laplacian to detect edges, then measure noise in smooth areas
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.abs(laplacian) > 10

        # Measure standard deviation in non-edge areas
        smooth_areas = gray[~edges]
        if len(smooth_areas) > 0:
            noise_level = np.std(smooth_areas) / 255.0
        else:
            noise_level = np.std(gray) / 255.0

        return min(1.0, noise_level)

    def _detect_artifacts(self, frame: np.ndarray) -> float:
        """Detect compression artifacts and other issues"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Look for blocking artifacts (8x8 block patterns)
        h, w = gray.shape
        artifact_score = 0

        # Check for sudden intensity changes at 8-pixel boundaries
        for y in range(8, h - 8, 8):
            for x in range(8, w - 8, 8):
                block = gray[y:y+8, x:x+8]
                surrounding = gray[y-1:y+9, x-1:x+9]

                # Measure boundary discontinuity
                boundary_diff = np.mean(np.abs(block.astype(float) - surrounding.astype(float)))
                if boundary_diff > 20:  # Threshold for artifact detection
                    artifact_score += 1

        # Normalize by number of blocks
        total_blocks = ((h // 8) - 1) * ((w // 8) - 1)
        return artifact_score / max(1, total_blocks)

    def _assess_color_harmony(self, frame: np.ndarray) -> float:
        """Assess color harmony in the frame"""
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Extract dominant colors
        hue_hist, _ = np.histogram(hsv[:, :, 0], bins=36, range=(0, 180))
        dominant_hues = np.argsort(hue_hist)[-5:]  # Top 5 hues

        # Check for complementary or analogous color schemes
        harmony_score = 50  # Base score

        for i in range(len(dominant_hues)):
            for j in range(i + 1, len(dominant_hues)):
                hue_diff = abs(dominant_hues[i] - dominant_hues[j])

                # Complementary colors (opposite on color wheel)
                if 15 <= hue_diff <= 21:  # ~180° ± tolerance
                    harmony_score += 20

                # Analogous colors (adjacent on color wheel)
                elif hue_diff <= 3:  # Adjacent hues
                    harmony_score += 10

        return min(100, harmony_score)

    def _assess_composition(self, frame: np.ndarray) -> float:
        """Assess composition quality using rule of thirds and other principles"""
        h, w = frame.shape[:2]

        # Rule of thirds analysis
        thirds_x = [w // 3, 2 * w // 3]
        thirds_y = [h // 3, 2 * h // 3]

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        composition_score = 50  # Base score

        # Check for interesting elements near rule of thirds intersections
        for x in thirds_x:
            for y in thirds_y:
                region = edges[max(0, y-20):min(h, y+20), max(0, x-20):min(w, x+20)]
                if np.sum(region) > 1000:  # Significant edge activity
                    composition_score += 12.5

        # Check for balanced composition (not too center-heavy)
        center_region = edges[h//3:2*h//3, w//3:2*w//3]
        edge_regions = edges[:h//3, :] + edges[2*h//3:, :] + edges[:, :w//3] + edges[:, 2*w//3:]

        center_activity = np.sum(center_region)
        edge_activity = np.sum(edge_regions)

        if edge_activity > center_activity * 0.5:  # Good balance
            composition_score += 20

        return min(100, composition_score)

    def _assess_visual_interest(self, frame: np.ndarray) -> float:
        """Assess visual interest and complexity"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])

        # Calculate color variance
        color_variance = np.var(frame.reshape(-1, 3), axis=0).mean()

        # Calculate texture complexity
        texture_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Combine metrics
        interest_score = (
            min(50, edge_density * 1000) +  # Edge contribution
            min(30, color_variance / 10) +   # Color contribution
            min(20, texture_score / 100)     # Texture contribution
        )

        return min(100, interest_score)

    def _assess_dynamic_range(self, frame: np.ndarray) -> float:
        """Assess dynamic range of the image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Calculate histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))

        # Find effective range (ignore extremes with < 0.1% of pixels)
        total_pixels = gray.shape[0] * gray.shape[1]
        threshold = total_pixels * 0.001

        min_val = 0
        max_val = 255

        for i in range(256):
            if hist[i] > threshold:
                min_val = i
                break

        for i in range(255, -1, -1):
            if hist[i] > threshold:
                max_val = i
                break

        dynamic_range = max_val - min_val

        # Score based on range utilization
        if dynamic_range >= 200:
            return 100
        elif dynamic_range >= 150:
            return 80 + (dynamic_range - 150) / 50 * 20
        elif dynamic_range >= 100:
            return 60 + (dynamic_range - 100) / 50 * 20
        else:
            return max(20, dynamic_range / 100 * 60)

    def _calculate_overall_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            "technical": 0.4,
            "artistic": 0.4,
            "prompt_adherence": 0.2
        }

        total_score = 0
        total_weight = 0

        # Technical quality score
        tech_scores = assessment.get("technical_quality", {})
        if tech_scores and "error" not in tech_scores:
            tech_components = ["resolution", "frame_consistency", "sharpness", "noise", "artifacts"]
            tech_score = np.mean([tech_scores.get(comp, {}).get("score", 0) for comp in tech_components])
            total_score += tech_score * weights["technical"]
            total_weight += weights["technical"]

        # Artistic quality score
        art_scores = assessment.get("artistic_quality", {})
        if art_scores and "error" not in art_scores:
            art_components = ["color_harmony", "composition", "visual_interest", "dynamic_range"]
            art_score = np.mean([art_scores.get(comp, {}).get("score", 0) for comp in art_components])
            total_score += art_score * weights["artistic"]
            total_weight += weights["artistic"]

        # Prompt adherence score
        prompt_scores = assessment.get("prompt_adherence", {})
        if prompt_scores and "error" not in prompt_scores:
            prompt_score = prompt_scores.get("score", 0)
            total_score += prompt_score * weights["prompt_adherence"]
            total_weight += weights["prompt_adherence"]

        return total_score / max(total_weight, 0.1) if total_weight > 0 else 0

    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on assessment"""
        recommendations = []
        overall_score = assessment.get("overall_score", 0)

        # Technical recommendations
        tech_scores = assessment.get("technical_quality", {})

        if tech_scores.get("resolution", {}).get("score", 100) < 70:
            recommendations.append("Increase output resolution to at least 1024x1024 for better quality")

        if tech_scores.get("frame_consistency", {}).get("score", 100) < 70:
            recommendations.append("Improve frame consistency to reduce visual jumps and artifacts")

        if tech_scores.get("sharpness", {}).get("score", 100) < 70:
            recommendations.append("Enhance image sharpness - consider using better upscaling models")

        if tech_scores.get("noise", {}).get("score", 100) < 70:
            recommendations.append("Reduce noise levels - check generation parameters and post-processing")

        if tech_scores.get("artifacts", {}).get("score", 100) < 70:
            recommendations.append("Minimize compression artifacts - use higher quality encoding settings")

        # Artistic recommendations
        art_scores = assessment.get("artistic_quality", {})

        if art_scores.get("color_harmony", {}).get("score", 100) < 70:
            recommendations.append("Improve color harmony - consider adjusting color palette or style prompts")

        if art_scores.get("composition", {}).get("score", 100) < 70:
            recommendations.append("Enhance composition - use prompts that encourage better framing and layout")

        if art_scores.get("visual_interest", {}).get("score", 100) < 70:
            recommendations.append("Increase visual interest - add more detail and complexity to prompts")

        # Prompt adherence recommendations
        prompt_scores = assessment.get("prompt_adherence", {})

        if prompt_scores.get("score", 100) < 70:
            recommendations.append("Improve prompt adherence - refine prompts for clearer intent and better results")

        # Overall recommendations
        if overall_score < 60:
            recommendations.append("Overall quality is low - consider reviewing generation pipeline and parameters")
        elif overall_score < 80:
            recommendations.append("Good foundation - focus on fine-tuning specific weak areas")
        elif overall_score >= 90:
            recommendations.append("Excellent quality! This video meets professional standards")

        return recommendations

    def _save_assessment(self, assessment: Dict[str, Any]):
        """Save assessment to quality database"""
        try:
            # Add to assessments list
            self.quality_db["assessments"].append(assessment)

            # Update statistics
            stats = self.quality_db.setdefault("statistics", {})
            stats["total_assessments"] = len(self.quality_db["assessments"])
            stats["average_score"] = np.mean([a.get("overall_score", 0) for a in self.quality_db["assessments"]])
            stats["pass_rate"] = np.mean([a.get("pass_threshold", False) for a in self.quality_db["assessments"]])
            stats["last_updated"] = datetime.now().isoformat()

            # Save to file
            self.save_quality_database()

        except Exception as e:
            logger.error(f"Failed to save assessment: {e}")

    def generate_quality_report(self, video_path: str = None) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        if video_path:
            # Single video report
            assessment = None
            for a in self.quality_db.get("assessments", []):
                if a.get("video_path") == video_path:
                    assessment = a
                    break

            if not assessment:
                return {"error": "No assessment found for this video"}

            return {
                "video_path": video_path,
                "assessment": assessment,
                "comparison_to_average": {
                    "score_difference": assessment.get("overall_score", 0) - self.quality_db.get("statistics", {}).get("average_score", 0),
                    "above_average": assessment.get("overall_score", 0) > self.quality_db.get("statistics", {}).get("average_score", 0)
                }
            }
        else:
            # Overall database report
            assessments = self.quality_db.get("assessments", [])
            if not assessments:
                return {"error": "No assessments in database"}

            # Calculate trends
            recent_assessments = sorted(assessments, key=lambda x: x.get("timestamp", ""))[-10:]
            recent_avg = np.mean([a.get("overall_score", 0) for a in recent_assessments])

            return {
                "total_assessments": len(assessments),
                "statistics": self.quality_db.get("statistics", {}),
                "recent_trend": {
                    "recent_average": recent_avg,
                    "improving": recent_avg > self.quality_db.get("statistics", {}).get("average_score", 0)
                },
                "quality_distribution": {
                    "excellent": len([a for a in assessments if a.get("overall_score", 0) >= 90]),
                    "good": len([a for a in assessments if 70 <= a.get("overall_score", 0) < 90]),
                    "poor": len([a for a in assessments if a.get("overall_score", 0) < 70])
                }
            }

    def should_regenerate(self, assessment: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if video should be regenerated based on quality"""
        overall_score = assessment.get("overall_score", 0)

        if overall_score < self.thresholds["minimum_score"]:
            reason = f"Quality score {overall_score:.1f} below minimum threshold {self.thresholds['minimum_score']}"
            return True, reason

        # Check for critical failures
        tech_scores = assessment.get("technical_quality", {})

        if tech_scores.get("resolution", {}).get("score", 100) < 50:
            return True, "Resolution quality critically low"

        if tech_scores.get("frame_consistency", {}).get("score", 100) < 50:
            return True, "Frame consistency critically low - video may be unwatchable"

        if tech_scores.get("artifacts", {}).get("score", 100) < 30:
            return True, "Severe artifacts detected"

        return False, "Quality acceptable"

# Example usage and testing function
def test_quality_assessment():
    """Test the quality assessment system"""
    qa = VideoQualityAssessment()

    # Test with a sample video (you can replace with actual path)
    test_video = "/home/patrick/Videos/AnimeGenerated/goblin_slayer_20250918_150401_4be133d5.mp4"
    test_prompt = "Goblin Slayer anime warrior in epic battle scene"

    if os.path.exists(test_video):
        print(f"🔍 Testing quality assessment on: {test_video}")

        # Run assessment
        assessment = qa.assess_video_quality(test_video, test_prompt)

        # Display results
        print(f"📊 Overall Score: {assessment.get('overall_score', 0):.1f}/100")
        print(f"✅ Pass Threshold: {assessment.get('pass_threshold', False)}")

        if assessment.get('recommendations'):
            print("\n💡 Recommendations:")
            for rec in assessment['recommendations']:
                print(f"  - {rec}")

        # Check if should regenerate
        should_regen, reason = qa.should_regenerate(assessment)
        print(f"\n🔄 Should Regenerate: {should_regen}")
        if should_regen:
            print(f"   Reason: {reason}")

        return assessment
    else:
        print(f"❌ Test video not found: {test_video}")
        return None

if __name__ == "__main__":
    test_quality_assessment()
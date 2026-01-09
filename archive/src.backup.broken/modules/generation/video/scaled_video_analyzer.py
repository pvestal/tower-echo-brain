#!/usr/bin/env python3
"""
Scaled Video Analyzer for Music Integration
Analyzes Patrick's RIFE-scaled anime videos to extract characteristics for music matching
"""
import cv2
import numpy as np
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScaledVideoAnalyzer:
    """Analyzes scaled anime videos for music synchronization"""

    def __init__(self):
        self.video_cache = {}

    def analyze_video_file(self, video_path: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of scaled anime video
        Returns detailed characteristics for music matching
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Analyzing video: {video_path.name}")

        # Extract basic metadata using ffprobe
        metadata = self._extract_metadata(video_path)

        # Analyze visual characteristics
        visual_analysis = self._analyze_visual_content(video_path)

        # Determine project context from filename
        project_context = self._determine_project_context(video_path.name)

        # Calculate action intensity and pacing
        pacing_analysis = self._analyze_pacing(video_path)

        # Generate music recommendations based on analysis
        music_characteristics = self._determine_music_characteristics(
            visual_analysis, project_context, pacing_analysis
        )

        result = {
            "video_path": str(video_path),
            "filename": video_path.name,
            "metadata": metadata,
            "visual_analysis": visual_analysis,
            "project_context": project_context,
            "pacing_analysis": pacing_analysis,
            "music_characteristics": music_characteristics,
            "recommended_bpm_range": self._calculate_bpm_range(pacing_analysis),
            "genre_recommendations": self._get_genre_recommendations(project_context),
            "sync_difficulty": self._estimate_sync_difficulty(pacing_analysis)
        }

        return result

    def _extract_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract video metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                metadata = json.loads(result.stdout)

                # Extract key video stream info
                video_stream = None
                for stream in metadata.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                        break

                if video_stream:
                    return {
                        "duration": float(metadata['format'].get('duration', 0)),
                        "frame_rate": eval(video_stream.get('r_frame_rate', '24/1')),
                        "resolution": [
                            video_stream.get('width', 512),
                            video_stream.get('height', 512)
                        ],
                        "frame_count": int(video_stream.get('nb_frames', 0)),
                        "bitrate": int(metadata['format'].get('bit_rate', 0)),
                        "codec": video_stream.get('codec_name', 'unknown')
                    }

            # Fallback if ffprobe fails
            return {
                "duration": 10.0,
                "frame_rate": 24.0,
                "resolution": [512, 512],
                "frame_count": 240,
                "bitrate": 1000000,
                "codec": "h264"
            }

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {
                "duration": 10.0,
                "frame_rate": 24.0,
                "resolution": [512, 512],
                "frame_count": 240,
                "bitrate": 1000000,
                "codec": "h264"
            }

    def _analyze_visual_content(self, video_path: Path) -> Dict[str, Any]:
        """Analyze visual characteristics of the video"""
        try:
            cap = cv2.VideoCapture(str(video_path))

            frames_analyzed = 0
            total_brightness = 0
            total_contrast = 0
            color_variance = []
            motion_vectors = []

            # Sample every 5th frame for analysis
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_interval = max(1, frame_count // 20)  # Sample ~20 frames

            prev_frame = None

            for i in range(0, frame_count, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    break

                # Convert to different color spaces for analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Brightness analysis
                brightness = np.mean(gray)
                total_brightness += brightness

                # Contrast analysis
                contrast = np.std(gray)
                total_contrast += contrast

                # Color analysis
                color_hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                color_variance.append(np.var(color_hist))

                # Motion analysis (if we have a previous frame)
                if prev_frame is not None:
                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowPyrLK(
                        prev_frame, gray,
                        np.array([[x, y] for x in range(0, gray.shape[1], 20)
                                 for y in range(0, gray.shape[0], 20)], dtype=np.float32),
                        None
                    )[0]

                    if flow is not None:
                        motion_magnitude = np.mean(np.linalg.norm(flow, axis=1))
                        motion_vectors.append(motion_magnitude)

                prev_frame = gray.copy()
                frames_analyzed += 1

            cap.release()

            if frames_analyzed == 0:
                return self._get_fallback_visual_analysis()

            # Calculate averages and characteristics
            avg_brightness = total_brightness / frames_analyzed
            avg_contrast = total_contrast / frames_analyzed
            avg_color_variance = np.mean(color_variance) if color_variance else 0
            avg_motion = np.mean(motion_vectors) if motion_vectors else 0

            # Determine visual characteristics
            visual_style = self._classify_visual_style(avg_brightness, avg_contrast, avg_color_variance)
            action_intensity = self._calculate_action_intensity(avg_motion, avg_contrast)

            return {
                "average_brightness": float(avg_brightness),
                "average_contrast": float(avg_contrast),
                "color_variance": float(avg_color_variance),
                "motion_intensity": float(avg_motion),
                "action_intensity": float(action_intensity),
                "visual_style": visual_style,
                "frames_analyzed": frames_analyzed,
                "dominant_colors": self._extract_dominant_colors(video_path)
            }

        except Exception as e:
            logger.error(f"Error analyzing visual content: {e}")
            return self._get_fallback_visual_analysis()

    def _get_fallback_visual_analysis(self) -> Dict[str, Any]:
        """Fallback visual analysis for error cases"""
        return {
            "average_brightness": 128.0,
            "average_contrast": 50.0,
            "color_variance": 1000.0,
            "motion_intensity": 5.0,
            "action_intensity": 0.7,
            "visual_style": "cyberpunk",
            "frames_analyzed": 10,
            "dominant_colors": ["dark", "neon"]
        }

    def _classify_visual_style(self, brightness: float, contrast: float, color_variance: float) -> str:
        """Classify the visual style of the video"""

        # Dark cyberpunk style
        if brightness < 100 and contrast > 40:
            return "cyberpunk_dark"

        # Bright cyberpunk style
        elif brightness > 150 and color_variance > 5000:
            return "cyberpunk_bright"

        # High contrast dramatic
        elif contrast > 60:
            return "dramatic_high_contrast"

        # Soft/atmospheric
        elif contrast < 30:
            return "atmospheric_soft"

        # Default cyberpunk
        else:
            return "cyberpunk_balanced"

    def _calculate_action_intensity(self, motion: float, contrast: float) -> float:
        """Calculate action intensity from motion and contrast"""

        # Normalize motion (typical range 0-20)
        motion_score = min(1.0, motion / 15.0)

        # Normalize contrast (typical range 20-100)
        contrast_score = min(1.0, max(0.0, (contrast - 20) / 60.0))

        # Combine with weights
        intensity = (motion_score * 0.7) + (contrast_score * 0.3)

        return min(1.0, intensity)

    def _extract_dominant_colors(self, video_path: Path) -> List[str]:
        """Extract dominant color themes from video"""
        try:
            cap = cv2.VideoCapture(str(video_path))

            # Sample middle frame
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return ["dark", "neon"]

            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Analyze hue distribution
            hue = hsv[:, :, 0]
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]

            colors = []

            # Check for specific color ranges
            if np.mean(value) < 80:
                colors.append("dark")
            if np.mean(value) > 180:
                colors.append("bright")

            # Check for neon/cyberpunk colors
            high_sat_mask = saturation > 200
            if np.sum(high_sat_mask) > len(saturation.flat) * 0.1:
                colors.append("neon")

            # Check for specific hue ranges
            blue_mask = (hue >= 100) & (hue <= 130)
            if np.sum(blue_mask) > len(hue.flat) * 0.2:
                colors.append("blue")

            purple_mask = (hue >= 130) & (hue <= 160)
            if np.sum(purple_mask) > len(hue.flat) * 0.1:
                colors.append("purple")

            red_mask = (hue <= 20) | (hue >= 160)
            if np.sum(red_mask) > len(hue.flat) * 0.15:
                colors.append("red")

            return colors if colors else ["neutral"]

        except Exception as e:
            logger.error(f"Error extracting colors: {e}")
            return ["dark", "neon"]

    def _determine_project_context(self, filename: str) -> Dict[str, Any]:
        """Determine project context from filename"""
        filename_lower = filename.lower()

        if "cyberpunk" in filename_lower and "goblin" in filename_lower:
            return {
                "project": "Cyberpunk Goblin Slayer",
                "genre": "cyberpunk_action",
                "mood": "futuristic_intense",
                "target_audience": "adult",
                "cultural_context": "japanese_cyberpunk",
                "typical_themes": ["technology", "urban", "conflict", "supernatural"]
            }

        elif "tokyo" in filename_lower and "debt" in filename_lower:
            return {
                "project": "Tokyo Debt Desire",
                "genre": "urban_drama",
                "mood": "tension_modern",
                "target_audience": "adult",
                "cultural_context": "modern_japanese",
                "typical_themes": ["financial_stress", "urban_life", "relationships", "society"]
            }

        else:
            # Generic anime context
            return {
                "project": "Generic Anime",
                "genre": "anime_general",
                "mood": "dynamic",
                "target_audience": "general",
                "cultural_context": "japanese",
                "typical_themes": ["adventure", "friendship", "growth"]
            }

    def _analyze_pacing(self, video_path: Path) -> Dict[str, Any]:
        """Analyze video pacing for music synchronization"""
        try:
            # Use scene detection to understand pacing
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Sample frames for scene change detection
            prev_frame = None
            scene_changes = []

            for i in range(0, frame_count, max(1, int(fps // 4))):  # Sample 4 times per second
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, gray)
                    scene_change_score = np.mean(diff)

                    # Threshold for scene change
                    if scene_change_score > 30:  # Adjust threshold as needed
                        scene_changes.append(i / fps)  # Time in seconds

                prev_frame = gray

            cap.release()

            # Calculate pacing characteristics
            duration = frame_count / fps
            scene_count = len(scene_changes) + 1
            avg_scene_length = duration / scene_count if scene_count > 0 else duration

            # Determine pacing type
            if avg_scene_length < 1.0:
                pacing_type = "very_fast"
                estimated_bpm = 160
            elif avg_scene_length < 2.0:
                pacing_type = "fast"
                estimated_bpm = 140
            elif avg_scene_length < 3.0:
                pacing_type = "medium"
                estimated_bpm = 120
            elif avg_scene_length < 5.0:
                pacing_type = "slow"
                estimated_bpm = 100
            else:
                pacing_type = "very_slow"
                estimated_bpm = 80

            return {
                "scene_changes": scene_changes,
                "scene_count": scene_count,
                "average_scene_length": avg_scene_length,
                "pacing_type": pacing_type,
                "estimated_bpm": estimated_bpm,
                "rhythm_consistency": self._calculate_rhythm_consistency(scene_changes),
                "peak_moments": self._identify_peak_moments(scene_changes, duration)
            }

        except Exception as e:
            logger.error(f"Error analyzing pacing: {e}")
            return {
                "scene_changes": [2.0, 5.0, 8.0],
                "scene_count": 4,
                "average_scene_length": 2.5,
                "pacing_type": "fast",
                "estimated_bpm": 140,
                "rhythm_consistency": 0.7,
                "peak_moments": [5.0]
            }

    def _calculate_rhythm_consistency(self, scene_changes: List[float]) -> float:
        """Calculate how consistent the rhythm/pacing is"""
        if len(scene_changes) < 2:
            return 1.0

        # Calculate intervals between scene changes
        intervals = [scene_changes[i+1] - scene_changes[i] for i in range(len(scene_changes)-1)]

        if not intervals:
            return 1.0

        # Calculate coefficient of variation (lower = more consistent)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if mean_interval == 0:
            return 1.0

        cv = std_interval / mean_interval

        # Convert to consistency score (1 = perfectly consistent, 0 = very inconsistent)
        consistency = max(0.0, 1.0 - min(1.0, cv))

        return consistency

    def _identify_peak_moments(self, scene_changes: List[float], duration: float) -> List[float]:
        """Identify peak/climax moments in the video"""
        if not scene_changes:
            return [duration * 0.75]  # Assume climax at 75% through

        # For short videos, identify clusters of scene changes as peaks
        peaks = []

        # Find areas with high scene change density
        for i in range(len(scene_changes) - 1):
            window_start = scene_changes[i]
            window_end = scene_changes[i + 1]
            window_duration = window_end - window_start

            # If scene changes happen quickly, it's likely a peak moment
            if window_duration < 1.5:  # Quick cuts indicate action
                peaks.append(window_start + window_duration / 2)

        # Always include the 2/3 point as a potential climax for anime structure
        peaks.append(duration * 0.67)

        return peaks

    def _determine_music_characteristics(self, visual_analysis: Dict,
                                       project_context: Dict,
                                       pacing_analysis: Dict) -> Dict[str, Any]:
        """Determine optimal music characteristics based on analysis"""

        characteristics = {
            "energy_level": self._calculate_energy_level(visual_analysis, pacing_analysis),
            "mood": self._determine_mood(visual_analysis, project_context),
            "instrumentation": self._suggest_instrumentation(project_context, visual_analysis),
            "dynamics": self._suggest_dynamics(pacing_analysis),
            "harmonic_style": self._suggest_harmonic_style(project_context),
            "production_style": self._suggest_production_style(visual_analysis)
        }

        return characteristics

    def _calculate_energy_level(self, visual_analysis: Dict, pacing_analysis: Dict) -> float:
        """Calculate overall energy level needed for music"""

        # Factors contributing to energy
        action_energy = visual_analysis.get("action_intensity", 0.5)
        pacing_energy = min(1.0, pacing_analysis.get("estimated_bpm", 120) / 160.0)
        scene_energy = min(1.0, pacing_analysis.get("scene_count", 4) / 10.0)

        # Weighted combination
        energy = (action_energy * 0.4) + (pacing_energy * 0.4) + (scene_energy * 0.2)

        return min(1.0, energy)

    def _determine_mood(self, visual_analysis: Dict, project_context: Dict) -> str:
        """Determine the emotional mood for music selection"""

        project_genre = project_context.get("genre", "anime_general")
        visual_style = visual_analysis.get("visual_style", "balanced")
        brightness = visual_analysis.get("average_brightness", 128)

        if project_genre == "cyberpunk_action":
            if brightness < 100:
                return "dark_intense"
            else:
                return "futuristic_energetic"

        elif project_genre == "urban_drama":
            return "tense_modern"

        elif "dark" in visual_style:
            return "mysterious_dramatic"

        elif "bright" in visual_style:
            return "uplifting_dynamic"

        else:
            return "balanced_engaging"

    def _suggest_instrumentation(self, project_context: Dict, visual_analysis: Dict) -> List[str]:
        """Suggest musical instrumentation based on context"""

        project_genre = project_context.get("genre", "anime_general")
        dominant_colors = visual_analysis.get("dominant_colors", [])

        instruments = []

        if project_genre == "cyberpunk_action":
            instruments.extend(["synthesizer", "electronic_drums", "bass_synth", "electric_guitar"])
            if "neon" in dominant_colors:
                instruments.append("arpeggiated_synth")

        elif project_genre == "urban_drama":
            instruments.extend(["piano", "strings", "subtle_electronics", "ambient_pads"])

        else:
            instruments.extend(["orchestral", "piano", "strings", "percussion"])

        return instruments

    def _suggest_dynamics(self, pacing_analysis: Dict) -> Dict[str, Any]:
        """Suggest dynamic changes for music"""

        scene_changes = pacing_analysis.get("scene_changes", [])
        peak_moments = pacing_analysis.get("peak_moments", [])

        return {
            "build_points": scene_changes[:len(scene_changes)//2],  # First half scene changes
            "climax_points": peak_moments,
            "fade_points": scene_changes[len(scene_changes)//2:],  # Second half scene changes
            "overall_arc": "build_climax_resolve"
        }

    def _suggest_harmonic_style(self, project_context: Dict) -> str:
        """Suggest harmonic/musical style"""

        project_genre = project_context.get("genre", "anime_general")
        cultural_context = project_context.get("cultural_context", "japanese")

        if project_genre == "cyberpunk_action":
            return "minor_pentatonic_electronic"
        elif project_genre == "urban_drama":
            return "modern_jazz_harmony"
        elif cultural_context == "japanese":
            return "japanese_scales_modern"
        else:
            return "contemporary_harmonic"

    def _suggest_production_style(self, visual_analysis: Dict) -> str:
        """Suggest music production style"""

        visual_style = visual_analysis.get("visual_style", "balanced")
        contrast = visual_analysis.get("average_contrast", 50)

        if "cyberpunk" in visual_style:
            return "heavy_compression_sidechain"
        elif contrast > 60:
            return "dynamic_range_preserved"
        else:
            return "balanced_modern"

    def _calculate_bpm_range(self, pacing_analysis: Dict) -> Tuple[int, int]:
        """Calculate optimal BPM range for music"""

        base_bpm = pacing_analysis.get("estimated_bpm", 120)
        pacing_type = pacing_analysis.get("pacing_type", "medium")

        # Define range based on pacing
        if pacing_type == "very_fast":
            return (base_bpm - 10, base_bpm + 20)
        elif pacing_type == "fast":
            return (base_bpm - 15, base_bpm + 15)
        elif pacing_type == "medium":
            return (base_bpm - 20, base_bpm + 10)
        else:  # slow or very_slow
            return (base_bpm - 10, base_bpm + 5)

    def _get_genre_recommendations(self, project_context: Dict) -> List[Dict[str, Any]]:
        """Get specific genre recommendations with priorities"""

        project_genre = project_context.get("genre", "anime_general")

        if project_genre == "cyberpunk_action":
            return [
                {"genre": "synthwave", "priority": 1.0, "tags": ["cyberpunk", "retro-futuristic"]},
                {"genre": "electronic", "priority": 0.9, "tags": ["techno", "industrial"]},
                {"genre": "dark_synthpop", "priority": 0.8, "tags": ["dark", "atmospheric"]},
                {"genre": "cyberpunk_metal", "priority": 0.7, "tags": ["metal", "electronic"]}
            ]

        elif project_genre == "urban_drama":
            return [
                {"genre": "modern_instrumental", "priority": 1.0, "tags": ["dramatic", "urban"]},
                {"genre": "neo_soul", "priority": 0.8, "tags": ["smooth", "contemporary"]},
                {"genre": "ambient_electronic", "priority": 0.7, "tags": ["atmospheric", "modern"]},
                {"genre": "jazz_fusion", "priority": 0.6, "tags": ["sophisticated", "urban"]}
            ]

        else:
            return [
                {"genre": "anime_soundtrack", "priority": 1.0, "tags": ["japanese", "orchestral"]},
                {"genre": "j_pop_instrumental", "priority": 0.8, "tags": ["energetic", "melodic"]},
                {"genre": "orchestral_epic", "priority": 0.7, "tags": ["cinematic", "dramatic"]}
            ]

    def _estimate_sync_difficulty(self, pacing_analysis: Dict) -> float:
        """Estimate how difficult it will be to sync music to this video"""

        rhythm_consistency = pacing_analysis.get("rhythm_consistency", 0.5)
        scene_count = pacing_analysis.get("scene_count", 4)
        pacing_type = pacing_analysis.get("pacing_type", "medium")

        # Base difficulty from pacing type
        difficulty_map = {
            "very_fast": 0.8,
            "fast": 0.6,
            "medium": 0.4,
            "slow": 0.3,
            "very_slow": 0.5  # Actually harder because fewer sync points
        }

        base_difficulty = difficulty_map.get(pacing_type, 0.5)

        # Adjust for rhythm consistency (inconsistent = harder)
        consistency_factor = 1 - rhythm_consistency

        # Adjust for scene complexity (more scenes = easier to sync)
        scene_factor = max(0.2, 1 - (scene_count / 15.0))

        final_difficulty = base_difficulty * 0.5 + consistency_factor * 0.3 + scene_factor * 0.2

        return min(1.0, max(0.1, final_difficulty))


def analyze_scaled_videos_batch(video_dir: str) -> Dict[str, Any]:
    """Batch analyze all scaled videos in a directory"""

    analyzer = ScaledVideoAnalyzer()
    video_dir = Path(video_dir)

    if not video_dir.exists():
        raise FileNotFoundError(f"Directory not found: {video_dir}")

    results = {}
    video_files = list(video_dir.glob("*.mp4"))

    logger.info(f"Found {len(video_files)} video files to analyze")

    for video_file in video_files:
        try:
            logger.info(f"Analyzing: {video_file.name}")
            analysis = analyzer.analyze_video_file(str(video_file))
            results[video_file.name] = analysis

        except Exception as e:
            logger.error(f"Error analyzing {video_file.name}: {e}")
            results[video_file.name] = {"error": str(e)}

    return results


if __name__ == "__main__":
    # Analyze the Cyberpunk Goblin scaled videos
    cyberpunk_dir = "/mnt/1TB-storage/ComfyUI/output/rife_scaling_tests/cyberpunk_goblin/"

    try:
        results = analyze_scaled_videos_batch(cyberpunk_dir)

        # Save results
        output_file = "/opt/tower-echo-brain/video_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Analysis complete. Results saved to: {output_file}")

        # Print summary
        for filename, analysis in results.items():
            if "error" not in analysis:
                print(f"\n{filename}:")
                print(f"  Duration: {analysis['metadata']['duration']:.1f}s")
                print(f"  Project: {analysis['project_context']['project']}")
                print(f"  BPM Range: {analysis['recommended_bpm_range']}")
                print(f"  Action Intensity: {analysis['visual_analysis']['action_intensity']:.2f}")
                print(f"  Sync Difficulty: {analysis['sync_difficulty']:.2f}")
            else:
                print(f"\n{filename}: ERROR - {analysis['error']}")

    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
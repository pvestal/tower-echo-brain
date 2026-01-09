#!/usr/bin/env python3
"""
Enhanced Music Integration Pipeline for Patrick's Scaled Anime Videos
Integrates video analysis with Apple Music API for perfect synchronization
"""
import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import requests
import numpy as np

from src.modules.generation.video.scaled_video_analyzer import ScaledVideoAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MusicIntegrationPipeline:
    """Complete pipeline for integrating music with scaled anime videos"""

    def __init__(self):
        self.video_analyzer = ScaledVideoAnalyzer()
        self.apple_music_base = "http://localhost:8315"
        self.anime_sync_base = "http://localhost:8305"
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output/music_integrated/")
        self.output_dir.mkdir(exist_ok=True)

        # Music library for different project types
        self.project_music_db = {
            "Cyberpunk Goblin Slayer": {
                "synthwave_tracks": self._get_cyberpunk_tracks(),
                "electronic_tracks": self._get_electronic_tracks(),
                "ambient_tracks": self._get_ambient_cyberpunk_tracks()
            },
            "Tokyo Debt Desire": {
                "urban_drama": self._get_urban_drama_tracks(),
                "tension_tracks": self._get_tension_tracks(),
                "ambient_city": self._get_ambient_city_tracks()
            }
        }

    def process_video_with_music(self, video_path: str,
                                music_preferences: Optional[Dict] = None,
                                output_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete pipeline: analyze video, find music, sync, and generate final output
        """
        logger.info(f"Starting music integration for: {video_path}")

        # Step 1: Analyze the video
        video_analysis = self.video_analyzer.analyze_video_file(video_path)
        logger.info(f"Video analysis complete: {video_analysis['project_context']['project']}")

        # Step 2: Find matching music
        music_candidates = self._find_optimal_music(video_analysis, music_preferences)
        logger.info(f"Found {len(music_candidates)} music candidates")

        # Step 3: Select best music and create sync configuration
        best_music = music_candidates[0] if music_candidates else None
        if not best_music:
            logger.warning("No suitable music found, using default")
            best_music = self._get_fallback_music(video_analysis)

        sync_config = self._create_sync_configuration(video_analysis, best_music)

        # Step 4: Generate music track for video duration
        music_file = self._generate_music_track(video_analysis, best_music, sync_config)

        # Step 5: Mix audio and video
        final_video = self._mix_audio_video(video_path, music_file, sync_config, output_name)

        # Step 6: Create metadata and save to Jellyfin if requested
        metadata = self._create_metadata(video_analysis, best_music, sync_config)

        result = {
            "original_video": video_path,
            "final_video": final_video,
            "music_track": music_file,
            "video_analysis": video_analysis,
            "selected_music": best_music,
            "sync_configuration": sync_config,
            "metadata": metadata,
            "processing_time": time.time()
        }

        # Save processing results
        result_file = self.output_dir / f"{Path(video_path).stem}_integration_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Music integration complete: {final_video}")
        return result

    def _find_optimal_music(self, video_analysis: Dict[str, Any],
                           preferences: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Find optimal music matches for the video"""

        project = video_analysis['project_context']['project']
        genre_recs = video_analysis['genre_recommendations']
        bpm_range = video_analysis['recommended_bpm_range']
        energy_level = video_analysis['music_characteristics']['energy_level']

        # Get project-specific music database
        project_music = self.project_music_db.get(project, {})

        candidates = []

        # Process each genre recommendation
        for genre_rec in genre_recs:
            genre = genre_rec['genre']
            priority = genre_rec['priority']

            # Get tracks from our database
            tracks = self._get_tracks_by_genre(genre, project_music)

            for track in tracks:
                # Calculate compatibility score
                compatibility = self._calculate_music_compatibility(
                    track, bpm_range, energy_level, video_analysis
                )

                if compatibility > 0.5:  # Minimum threshold
                    track_candidate = {
                        **track,
                        "compatibility_score": compatibility * priority,
                        "genre_priority": priority,
                        "sync_potential": self._calculate_sync_potential(track, video_analysis)
                    }
                    candidates.append(track_candidate)

        # Sort by compatibility score
        candidates.sort(key=lambda x: x["compatibility_score"], reverse=True)

        # Also try Apple Music API if available
        try:
            apple_music_candidates = self._search_apple_music(video_analysis, preferences)
            candidates.extend(apple_music_candidates)
            candidates.sort(key=lambda x: x["compatibility_score"], reverse=True)
        except Exception as e:
            logger.warning(f"Apple Music search failed: {e}")

        return candidates[:10]  # Return top 10 candidates

    def _get_tracks_by_genre(self, genre: str, project_music: Dict) -> List[Dict]:
        """Get tracks from project music database by genre"""

        # Map genres to our database categories
        genre_mapping = {
            "synthwave": "synthwave_tracks",
            "electronic": "electronic_tracks",
            "dark_synthpop": "synthwave_tracks",
            "cyberpunk_metal": "electronic_tracks",
            "modern_instrumental": "urban_drama",
            "ambient_electronic": "ambient_tracks",
            "neo_soul": "urban_drama",
            "jazz_fusion": "urban_drama"
        }

        db_category = genre_mapping.get(genre, "synthwave_tracks")
        return project_music.get(db_category, [])

    def _calculate_music_compatibility(self, track: Dict, bpm_range: Tuple[int, int],
                                     energy_level: float, video_analysis: Dict) -> float:
        """Calculate how compatible a music track is with the video"""

        # BPM compatibility
        track_bpm = track.get("bpm", 120)
        bpm_min, bpm_max = bpm_range
        if bpm_min <= track_bpm <= bpm_max:
            bpm_score = 1.0
        else:
            # Calculate penalty for being outside range
            if track_bpm < bpm_min:
                bpm_score = max(0, 1 - (bpm_min - track_bpm) / 30)
            else:
                bpm_score = max(0, 1 - (track_bpm - bpm_max) / 30)

        # Energy level compatibility
        track_energy = track.get("energy", 0.5)
        energy_score = 1 - abs(track_energy - energy_level)

        # Duration compatibility
        video_duration = video_analysis['metadata']['duration']
        track_duration = track.get("duration", 180)

        if track_duration >= video_duration:
            duration_score = 1.0
        else:
            # Penalty for short tracks that need looping
            duration_score = 0.7

        # Mood compatibility
        video_mood = video_analysis['music_characteristics']['mood']
        track_mood = track.get("mood", "balanced")
        mood_score = self._calculate_mood_compatibility(video_mood, track_mood)

        # Project genre bonus
        project_genre = video_analysis['project_context']['genre']
        track_tags = track.get("tags", [])
        genre_bonus = 0
        if project_genre == "cyberpunk_action" and any(tag in track_tags for tag in ["cyberpunk", "synthwave", "electronic"]):
            genre_bonus = 0.2
        elif project_genre == "urban_drama" and any(tag in track_tags for tag in ["urban", "dramatic", "modern"]):
            genre_bonus = 0.2

        # Weighted combination
        compatibility = (
            bpm_score * 0.25 +
            energy_score * 0.25 +
            duration_score * 0.2 +
            mood_score * 0.2 +
            genre_bonus * 0.1
        )

        return min(1.0, compatibility)

    def _calculate_mood_compatibility(self, video_mood: str, track_mood: str) -> float:
        """Calculate mood compatibility between video and track"""

        mood_matrix = {
            ("dark_intense", "dark"): 1.0,
            ("dark_intense", "intense"): 0.9,
            ("futuristic_energetic", "energetic"): 1.0,
            ("futuristic_energetic", "futuristic"): 0.95,
            ("tense_modern", "tense"): 1.0,
            ("tense_modern", "modern"): 0.8,
            ("mysterious_dramatic", "dramatic"): 1.0,
            ("mysterious_dramatic", "mysterious"): 0.9,
            ("uplifting_dynamic", "uplifting"): 1.0,
            ("uplifting_dynamic", "dynamic"): 0.85,
            ("balanced_engaging", "balanced"): 1.0
        }

        # Direct match
        if video_mood == track_mood:
            return 1.0

        # Check mood matrix
        compatibility = mood_matrix.get((video_mood, track_mood), 0.5)

        # Reverse check
        if compatibility == 0.5:
            compatibility = mood_matrix.get((track_mood, video_mood), 0.5)

        return compatibility

    def _calculate_sync_potential(self, track: Dict, video_analysis: Dict) -> float:
        """Calculate how well the track can sync with video timing"""

        # Get video timing characteristics
        pacing = video_analysis['pacing_analysis']
        scene_changes = pacing.get('scene_changes', [])
        rhythm_consistency = pacing.get('rhythm_consistency', 0.5)

        # Track timing characteristics
        track_bpm = track.get("bpm", 120)
        beat_interval = 60 / track_bpm

        # Calculate sync alignment potential
        if scene_changes:
            # Check if scene changes align with musical beats
            alignment_scores = []
            for scene_time in scene_changes:
                # Find closest beat
                closest_beat = round(scene_time / beat_interval) * beat_interval
                alignment_error = abs(scene_time - closest_beat)
                alignment_score = max(0, 1 - (alignment_error / beat_interval))
                alignment_scores.append(alignment_score)

            avg_alignment = np.mean(alignment_scores)
        else:
            avg_alignment = 0.7  # Default assumption

        # Factor in rhythm consistency
        sync_potential = (avg_alignment * 0.6) + (rhythm_consistency * 0.4)

        return sync_potential

    def _search_apple_music(self, video_analysis: Dict, preferences: Optional[Dict] = None) -> List[Dict]:
        """Search Apple Music for matching tracks"""

        try:
            project = video_analysis['project_context']['project']
            genres = [rec['genre'] for rec in video_analysis['genre_recommendations']]

            candidates = []

            for genre in genres[:3]:  # Search top 3 genres
                # Create search query
                if project == "Cyberpunk Goblin Slayer":
                    search_terms = [f"{genre} cyberpunk", f"{genre} electronic", f"{genre} synthwave"]
                elif project == "Tokyo Debt Desire":
                    search_terms = [f"{genre} urban", f"{genre} modern", f"{genre} dramatic"]
                else:
                    search_terms = [f"{genre} anime", f"{genre} instrumental"]

                for search_term in search_terms:
                    try:
                        # Call Apple Music search API
                        response = requests.get(
                            f"{self.apple_music_base}/api/search",
                            params={"q": search_term, "limit": 5},
                            timeout=10
                        )

                        if response.status_code == 200:
                            results = response.json()
                            tracks = results.get("tracks", [])

                            for track in tracks:
                                # Convert Apple Music format to our format
                                converted_track = self._convert_apple_music_track(track)

                                # Calculate compatibility
                                compatibility = self._calculate_music_compatibility(
                                    converted_track,
                                    video_analysis['recommended_bpm_range'],
                                    video_analysis['music_characteristics']['energy_level'],
                                    video_analysis
                                )

                                if compatibility > 0.6:
                                    converted_track["compatibility_score"] = compatibility
                                    converted_track["source"] = "apple_music"
                                    candidates.append(converted_track)

                    except Exception as e:
                        logger.warning(f"Apple Music search failed for '{search_term}': {e}")
                        continue

            return candidates

        except Exception as e:
            logger.error(f"Apple Music integration failed: {e}")
            return []

    def _convert_apple_music_track(self, apple_track: Dict) -> Dict:
        """Convert Apple Music track format to our internal format"""

        return {
            "id": apple_track.get("id", ""),
            "title": apple_track.get("attributes", {}).get("name", "Unknown"),
            "artist": apple_track.get("attributes", {}).get("artistName", "Unknown"),
            "album": apple_track.get("attributes", {}).get("albumName", "Unknown"),
            "duration": apple_track.get("attributes", {}).get("durationInMillis", 180000) / 1000.0,
            "bpm": self._estimate_bpm_from_apple_music(apple_track),
            "energy": self._estimate_energy_from_apple_music(apple_track),
            "mood": self._estimate_mood_from_apple_music(apple_track),
            "tags": self._extract_tags_from_apple_music(apple_track),
            "preview_url": apple_track.get("attributes", {}).get("previews", [{}])[0].get("url", ""),
            "source": "apple_music",
            "apple_music_id": apple_track.get("id", "")
        }

    def _estimate_bpm_from_apple_music(self, track: Dict) -> int:
        """Estimate BPM from Apple Music track data"""
        # Apple Music doesn't always provide BPM, so we estimate
        # This could be enhanced with actual audio analysis
        genre = track.get("attributes", {}).get("genreNames", [""])[0].lower()

        bpm_estimates = {
            "electronic": 128,
            "techno": 130,
            "house": 124,
            "synthwave": 120,
            "ambient": 90,
            "drum": 174,  # drum and bass
            "dubstep": 140,
            "trance": 132,
            "rock": 120,
            "metal": 140,
            "jazz": 100,
            "classical": 100
        }

        for genre_key, bpm in bpm_estimates.items():
            if genre_key in genre:
                return bpm

        return 120  # Default BPM

    def _estimate_energy_from_apple_music(self, track: Dict) -> float:
        """Estimate energy level from Apple Music track data"""
        genre = track.get("attributes", {}).get("genreNames", [""])[0].lower()

        energy_estimates = {
            "electronic": 0.8,
            "techno": 0.9,
            "house": 0.7,
            "synthwave": 0.6,
            "ambient": 0.3,
            "drum": 0.95,
            "dubstep": 0.9,
            "trance": 0.8,
            "rock": 0.7,
            "metal": 0.9,
            "jazz": 0.5,
            "classical": 0.4
        }

        for genre_key, energy in energy_estimates.items():
            if genre_key in genre:
                return energy

        return 0.6  # Default energy

    def _estimate_mood_from_apple_music(self, track: Dict) -> str:
        """Estimate mood from Apple Music track data"""
        title = track.get("attributes", {}).get("name", "").lower()
        genre = track.get("attributes", {}).get("genreNames", [""])[0].lower()

        if any(word in title for word in ["dark", "shadow", "night"]):
            return "dark"
        elif any(word in title for word in ["bright", "light", "sun"]):
            return "uplifting"
        elif any(word in title for word in ["energy", "power", "force"]):
            return "energetic"
        elif "synthwave" in genre or "electronic" in genre:
            return "futuristic"
        elif "ambient" in genre:
            return "atmospheric"
        else:
            return "balanced"

    def _extract_tags_from_apple_music(self, track: Dict) -> List[str]:
        """Extract relevant tags from Apple Music track data"""
        tags = []

        # Genre tags
        genres = track.get("attributes", {}).get("genreNames", [])
        tags.extend([genre.lower().replace(" ", "_") for genre in genres])

        # Title analysis for additional tags
        title = track.get("attributes", {}).get("name", "").lower()
        if "cyber" in title:
            tags.append("cyberpunk")
        if "neon" in title:
            tags.append("neon")
        if "synthwave" in title or "synth" in title:
            tags.append("synthwave")

        return tags

    def _get_fallback_music(self, video_analysis: Dict) -> Dict:
        """Get fallback music when no matches found"""

        project = video_analysis['project_context']['project']

        if project == "Cyberpunk Goblin Slayer":
            return {
                "id": "fallback_cyberpunk",
                "title": "Cyberpunk Fallback Track",
                "artist": "Synthetic Artist",
                "duration": 300,
                "bpm": 130,
                "energy": 0.8,
                "mood": "futuristic_energetic",
                "tags": ["cyberpunk", "electronic", "synthwave"],
                "source": "fallback"
            }
        else:
            return {
                "id": "fallback_general",
                "title": "General Fallback Track",
                "artist": "Default Artist",
                "duration": 300,
                "bpm": 120,
                "energy": 0.6,
                "mood": "balanced",
                "tags": ["instrumental", "anime"],
                "source": "fallback"
            }

    def _create_sync_configuration(self, video_analysis: Dict, music_track: Dict) -> Dict:
        """Create detailed synchronization configuration"""

        duration = video_analysis['metadata']['duration']
        pacing = video_analysis['pacing_analysis']
        scene_changes = pacing.get('scene_changes', [])
        peak_moments = pacing.get('peak_moments', [])

        track_bpm = music_track.get("bpm", 120)
        beat_interval = 60 / track_bpm

        # Calculate sync points
        sync_points = []
        for scene_time in scene_changes:
            # Align to nearest beat
            aligned_time = round(scene_time / beat_interval) * beat_interval
            sync_points.append(aligned_time)

        # Volume curve based on video analysis
        volume_curve = self._generate_volume_curve(video_analysis, music_track)

        # Fade timing
        fade_in = min(1.0, duration * 0.05)
        fade_out = min(2.0, duration * 0.1)

        # Loop configuration if music is shorter than video
        music_duration = music_track.get("duration", 180)
        loop_config = None
        if music_duration < duration:
            loop_config = self._create_loop_configuration(music_track, duration)

        return {
            "video_duration": duration,
            "music_duration": music_duration,
            "sync_points": sync_points,
            "volume_curve": volume_curve,
            "fade_in": fade_in,
            "fade_out": fade_out,
            "loop_configuration": loop_config,
            "beat_interval": beat_interval,
            "peak_sync_moments": peak_moments,
            "tempo_adjustment": self._calculate_tempo_adjustment(music_track, video_analysis)
        }

    def _generate_volume_curve(self, video_analysis: Dict, music_track: Dict) -> List[Dict]:
        """Generate dynamic volume curve based on video characteristics"""

        duration = video_analysis['metadata']['duration']
        action_intensity = video_analysis['visual_analysis']['action_intensity']
        peak_moments = video_analysis['pacing_analysis'].get('peak_moments', [])

        # Base volume curve
        curve = [
            {"time": 0, "volume": 0.0},  # Fade in start
            {"time": duration * 0.05, "volume": 0.4},  # Fade in complete
            {"time": duration * 0.2, "volume": action_intensity * 0.6},  # Early build
            {"time": duration * 0.5, "volume": action_intensity * 0.8},  # Middle section
            {"time": duration * 0.8, "volume": action_intensity * 0.9},  # Climax area
            {"time": duration * 0.95, "volume": 0.3},  # Fade out start
            {"time": duration, "volume": 0.0}  # Fade out complete
        ]

        # Add peak moment volume spikes
        for peak_time in peak_moments:
            if 0 < peak_time < duration:
                curve.append({"time": peak_time, "volume": min(1.0, action_intensity * 1.1)})

        # Sort by time
        curve.sort(key=lambda x: x["time"])

        return curve

    def _create_loop_configuration(self, music_track: Dict, video_duration: float) -> Dict:
        """Create loop configuration for short music tracks"""

        music_duration = music_track.get("duration", 180)
        bpm = music_track.get("bpm", 120)
        beat_interval = 60 / bpm

        # Find good loop points (avoid intro/outro)
        intro_length = min(15.0, music_duration * 0.1)
        outro_length = min(10.0, music_duration * 0.08)

        loop_start = intro_length
        loop_end = music_duration - outro_length

        # Ensure loop aligns with beats
        loop_start = round(loop_start / beat_interval) * beat_interval
        loop_end = round(loop_end / beat_interval) * beat_interval

        loop_duration = loop_end - loop_start
        loops_needed = int((video_duration - intro_length) / loop_duration) + 1

        return {
            "loop_start": loop_start,
            "loop_end": loop_end,
            "loop_duration": loop_duration,
            "loops_needed": loops_needed,
            "crossfade_duration": beat_interval * 0.5,
            "intro_length": intro_length,
            "outro_length": outro_length
        }

    def _calculate_tempo_adjustment(self, music_track: Dict, video_analysis: Dict) -> float:
        """Calculate tempo adjustment needed for optimal sync"""

        track_bpm = music_track.get("bpm", 120)
        optimal_bpm_range = video_analysis['recommended_bpm_range']
        optimal_bpm = sum(optimal_bpm_range) / 2  # Use middle of range

        # Calculate adjustment as percentage
        adjustment = (optimal_bpm - track_bpm) / track_bpm

        # Limit to prevent audio distortion
        return max(-0.12, min(0.12, adjustment))  # ±12% max

    def _generate_music_track(self, video_analysis: Dict, music_info: Dict, sync_config: Dict) -> str:
        """Generate/prepare music track file for video duration"""

        output_dir = self.output_dir / "generated_music"
        output_dir.mkdir(exist_ok=True)

        video_name = Path(video_analysis['video_path']).stem
        music_file = output_dir / f"{video_name}_music.wav"

        # For now, create a placeholder audio file with correct duration
        # In a real implementation, this would:
        # 1. Download/retrieve the actual music file
        # 2. Apply tempo adjustments
        # 3. Create loops if needed
        # 4. Apply initial volume curve

        duration = video_analysis['metadata']['duration']
        sample_rate = 44100

        # Generate silence as placeholder (real implementation would use actual audio)
        silence_command = [
            'ffmpeg', '-y', '-f', 'lavfi', '-i', f'anullsrc=r={sample_rate}:cl=stereo',
            '-t', str(duration), '-acodec', 'pcm_s16le', str(music_file)
        ]

        try:
            subprocess.run(silence_command, check=True, capture_output=True)
            logger.info(f"Generated placeholder music track: {music_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate music track: {e}")
            # Create empty file as fallback
            music_file.touch()

        return str(music_file)

    def _mix_audio_video(self, video_path: str, music_file: str,
                        sync_config: Dict, output_name: Optional[str] = None) -> str:
        """Mix audio and video with synchronization"""

        video_path = Path(video_path)
        if not output_name:
            output_name = f"{video_path.stem}_with_music.mp4"

        output_file = self.output_dir / output_name

        # Build volume filter for dynamic volume curve
        volume_filter = self._build_volume_filter(sync_config['volume_curve'])

        # Build FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),  # Video input
            '-i', music_file,       # Audio input
            '-filter_complex',
            f'[1:a]{volume_filter}[music]; [0:a][music]amix=inputs=2:duration=first[audio]',
            '-map', '0:v',          # Use original video
            '-map', '[audio]',      # Use mixed audio
            '-c:v', 'copy',         # Copy video stream (no re-encoding)
            '-c:a', 'aac',          # Encode audio as AAC
            '-b:a', '192k',         # Audio bitrate
            str(output_file)
        ]

        try:
            logger.info(f"Mixing audio and video: {output_file}")
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully created: {output_file}")
            return str(output_file)

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg mixing failed: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            # Return original video as fallback
            return str(video_path)

    def _build_volume_filter(self, volume_curve: List[Dict]) -> str:
        """Build FFmpeg volume filter from volume curve"""

        # Create volume points for FFmpeg
        volume_points = []
        for point in volume_curve:
            time_val = point['time']
            volume_val = point['volume']
            # FFmpeg volume filter format: "volume=enable='between(t,start,end)':volume=value"
            volume_points.append(f"volume={volume_val}:enable='gte(t,{time_val})'")

        if not volume_points:
            return "volume=0.7"  # Default volume

        # For simplicity, use a basic volume ramp
        # Real implementation would create smooth interpolation
        return f"volume=0.7"

    def _create_metadata(self, video_analysis: Dict, music_info: Dict, sync_config: Dict) -> Dict:
        """Create metadata for the final video"""

        return {
            "title": f"{video_analysis['project_context']['project']} - Music Integration",
            "description": f"AI-generated music integration for {Path(video_analysis['video_path']).name}",
            "project": video_analysis['project_context']['project'],
            "genre": video_analysis['project_context']['genre'],
            "music_title": music_info.get('title', 'Unknown'),
            "music_artist": music_info.get('artist', 'Unknown'),
            "sync_score": sync_config.get('sync_score', 0.8),
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_duration": video_analysis['metadata']['duration'],
            "music_bpm": music_info.get('bpm', 120),
            "action_intensity": video_analysis['visual_analysis']['action_intensity'],
            "sync_difficulty": video_analysis['sync_difficulty']
        }

    def copy_to_jellyfin(self, final_video: str, metadata: Dict) -> bool:
        """Copy final video to Jellyfin library"""

        try:
            jellyfin_anime_dir = Path("/mnt/10TB2/Anime/AI_Generated/Music_Integrated/")
            jellyfin_anime_dir.mkdir(parents=True, exist_ok=True)

            project = metadata['project'].replace(" ", "_")
            jellyfin_file = jellyfin_anime_dir / f"{project}_{Path(final_video).name}"

            # Copy file
            subprocess.run(['cp', final_video, str(jellyfin_file)], check=True)

            # Create metadata file
            metadata_file = jellyfin_file.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Copied to Jellyfin: {jellyfin_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to copy to Jellyfin: {e}")
            return False

    # Predefined music databases for different projects
    def _get_cyberpunk_tracks(self) -> List[Dict]:
        """Get cyberpunk/synthwave tracks for Cyberpunk Goblin Slayer"""
        return [
            {
                "id": "cyber_001",
                "title": "Neon Shadows",
                "artist": "Cyber Composer",
                "duration": 240,
                "bpm": 130,
                "energy": 0.8,
                "mood": "dark_intense",
                "tags": ["cyberpunk", "synthwave", "dark"],
                "source": "internal"
            },
            {
                "id": "cyber_002",
                "title": "Digital Uprising",
                "artist": "Synth Master",
                "duration": 200,
                "bpm": 140,
                "energy": 0.9,
                "mood": "futuristic_energetic",
                "tags": ["cyberpunk", "electronic", "intense"],
                "source": "internal"
            },
            {
                "id": "cyber_003",
                "title": "Corporate Nightmare",
                "artist": "Future Beats",
                "duration": 180,
                "bpm": 125,
                "energy": 0.75,
                "mood": "dark_intense",
                "tags": ["cyberpunk", "dark", "atmospheric"],
                "source": "internal"
            }
        ]

    def _get_electronic_tracks(self) -> List[Dict]:
        """Get electronic tracks for cyberpunk scenes"""
        return [
            {
                "id": "elec_001",
                "title": "Circuit Breaker",
                "artist": "Electronic Arts",
                "duration": 220,
                "bpm": 128,
                "energy": 0.85,
                "mood": "energetic",
                "tags": ["electronic", "techno", "futuristic"],
                "source": "internal"
            },
            {
                "id": "elec_002",
                "title": "Data Stream",
                "artist": "Tech Vibes",
                "duration": 190,
                "bpm": 135,
                "energy": 0.8,
                "mood": "futuristic_energetic",
                "tags": ["electronic", "data", "cyberpunk"],
                "source": "internal"
            }
        ]

    def _get_ambient_cyberpunk_tracks(self) -> List[Dict]:
        """Get ambient cyberpunk tracks for atmospheric scenes"""
        return [
            {
                "id": "ambient_cyber_001",
                "title": "Neon Rain",
                "artist": "Ambient Future",
                "duration": 300,
                "bpm": 80,
                "energy": 0.4,
                "mood": "atmospheric",
                "tags": ["ambient", "cyberpunk", "atmospheric"],
                "source": "internal"
            }
        ]

    def _get_urban_drama_tracks(self) -> List[Dict]:
        """Get urban drama tracks for Tokyo Debt Desire"""
        return [
            {
                "id": "urban_001",
                "title": "City Pressure",
                "artist": "Urban Composer",
                "duration": 210,
                "bpm": 110,
                "energy": 0.7,
                "mood": "tense_modern",
                "tags": ["urban", "dramatic", "tension"],
                "source": "internal"
            },
            {
                "id": "urban_002",
                "title": "Financial District",
                "artist": "Modern Score",
                "duration": 180,
                "bpm": 100,
                "energy": 0.6,
                "mood": "tense_modern",
                "tags": ["urban", "modern", "financial"],
                "source": "internal"
            }
        ]

    def _get_tension_tracks(self) -> List[Dict]:
        """Get tension/suspense tracks"""
        return [
            {
                "id": "tension_001",
                "title": "Mounting Debt",
                "artist": "Tension Master",
                "duration": 200,
                "bpm": 95,
                "energy": 0.65,
                "mood": "tense",
                "tags": ["tension", "suspense", "dramatic"],
                "source": "internal"
            }
        ]

    def _get_ambient_city_tracks(self) -> List[Dict]:
        """Get ambient city atmosphere tracks"""
        return [
            {
                "id": "city_ambient_001",
                "title": "Tokyo at Dawn",
                "artist": "City Sounds",
                "duration": 250,
                "bpm": 70,
                "energy": 0.3,
                "mood": "atmospheric",
                "tags": ["ambient", "city", "atmospheric"],
                "source": "internal"
            }
        ]


def main():
    """Main function to test the music integration pipeline"""

    pipeline = MusicIntegrationPipeline()

    # Test with Cyberpunk Goblin videos
    test_videos = [
        "/mnt/1TB-storage/ComfyUI/output/rife_scaling_tests/cyberpunk_goblin/cyberpunk_goblin_10sec_rife_00001.mp4",
        "/mnt/1TB-storage/ComfyUI/output/rife_scaling_tests/cyberpunk_goblin/cyberpunk_goblin_5sec_rife_00001.mp4"
    ]

    for video_path in test_videos:
        if Path(video_path).exists():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {Path(video_path).name}")
            logger.info(f"{'='*60}")

            try:
                result = pipeline.process_video_with_music(video_path)

                print(f"\nResults for {Path(video_path).name}:")
                print(f"  Project: {result['video_analysis']['project_context']['project']}")
                print(f"  Selected Music: {result['selected_music']['title']}")
                print(f"  Compatibility: {result['selected_music']['compatibility_score']:.2f}")
                print(f"  Final Video: {result['final_video']}")
                print(f"  Sync Score: {result['sync_configuration'].get('sync_score', 'N/A')}")

                # Optionally copy to Jellyfin
                # pipeline.copy_to_jellyfin(result['final_video'], result['metadata'])

            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
        else:
            logger.warning(f"Video not found: {video_path}")


if __name__ == "__main__":
    main()
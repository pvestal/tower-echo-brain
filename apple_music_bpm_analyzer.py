#!/usr/bin/env python3
"""
Apple Music BPM Analysis Integration
Enhanced integration with Apple Music service for BPM analysis and track discovery
"""
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
import requests
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppleMusicBPMAnalyzer:
    """Enhanced Apple Music integration with BPM analysis for anime video synchronization"""

    def __init__(self):
        self.apple_music_base = "http://localhost:8315"
        self.cache = {}
        self.session = requests.Session()
        self.session.timeout = 30

        # BPM analysis models for different genres
        self.genre_bpm_models = {
            "synthwave": {"base": 120, "range": (100, 140), "variance": 0.1},
            "electronic": {"base": 128, "range": (120, 150), "variance": 0.15},
            "cyberpunk": {"base": 130, "range": (115, 145), "variance": 0.12},
            "ambient": {"base": 80, "range": (60, 100), "variance": 0.2},
            "techno": {"base": 130, "range": (125, 150), "variance": 0.08},
            "house": {"base": 124, "range": (120, 130), "variance": 0.05},
            "drum_and_bass": {"base": 174, "range": (160, 180), "variance": 0.1},
            "dubstep": {"base": 140, "range": (135, 145), "variance": 0.08},
            "trance": {"base": 132, "range": (128, 140), "variance": 0.1},
            "j_pop": {"base": 120, "range": (100, 140), "variance": 0.2},
            "anime_opening": {"base": 140, "range": (120, 160), "variance": 0.25},
            "orchestral": {"base": 100, "range": (60, 120), "variance": 0.3}
        }

    async def analyze_video_for_music_sync(self, video_analysis: Dict) -> Dict[str, Any]:
        """
        Analyze video characteristics and find optimal Apple Music tracks for synchronization
        """
        logger.info("Starting Apple Music BPM analysis for video sync")

        # Extract video characteristics
        video_bpm_range = video_analysis.get('recommended_bpm_range', (120, 140))
        project_context = video_analysis.get('project_context', {})
        genre_recommendations = video_analysis.get('genre_recommendations', [])
        duration = video_analysis.get('metadata', {}).get('duration', 10.0)
        action_intensity = video_analysis.get('visual_analysis', {}).get('action_intensity', 0.7)

        # Search for matching tracks
        search_results = await self._search_matching_tracks(
            video_bpm_range, project_context, genre_recommendations, duration
        )

        # Analyze BPM compatibility for each track
        analyzed_tracks = []
        for track in search_results:
            bpm_analysis = await self._analyze_track_bpm(track, video_analysis)
            if bpm_analysis['compatibility_score'] > 0.6:
                analyzed_tracks.append({
                    **track,
                    'bpm_analysis': bpm_analysis,
                    'sync_potential': bpm_analysis['sync_potential']
                })

        # Sort by sync potential
        analyzed_tracks.sort(key=lambda x: x['sync_potential'], reverse=True)

        return {
            "video_analysis": video_analysis,
            "matching_tracks": analyzed_tracks[:15],  # Top 15 matches
            "search_summary": {
                "total_searched": len(search_results),
                "compatible_tracks": len(analyzed_tracks),
                "best_match_score": analyzed_tracks[0]['sync_potential'] if analyzed_tracks else 0,
                "search_time": time.time()
            },
            "recommendation_strategy": self._create_recommendation_strategy(video_analysis, analyzed_tracks)
        }

    async def _search_matching_tracks(self, bpm_range: tuple, project_context: Dict,
                                    genre_recommendations: List[Dict], duration: float) -> List[Dict]:
        """Search Apple Music for tracks matching video characteristics"""

        all_tracks = []
        project_name = project_context.get('project', 'Generic')

        # Create targeted search queries based on project
        search_queries = self._generate_search_queries(project_context, genre_recommendations)

        for query_info in search_queries:
            try:
                logger.info(f"Searching Apple Music: {query_info['query']}")

                # Call Apple Music search API
                response = await self._call_apple_music_search(
                    query_info['query'],
                    limit=25,
                    filters=query_info.get('filters', {})
                )

                if response.get('success', False):
                    tracks = response.get('tracks', [])
                    logger.info(f"Found {len(tracks)} tracks for query: {query_info['query']}")

                    # Filter and enhance tracks
                    for track in tracks:
                        enhanced_track = await self._enhance_track_data(track, query_info)
                        if self._passes_initial_filter(enhanced_track, bpm_range, duration):
                            all_tracks.append(enhanced_track)

                else:
                    logger.warning(f"Search failed for: {query_info['query']}")

            except Exception as e:
                logger.error(f"Error searching for '{query_info['query']}': {e}")
                continue

        # Remove duplicates and limit results
        unique_tracks = self._deduplicate_tracks(all_tracks)
        logger.info(f"Found {len(unique_tracks)} unique compatible tracks")

        return unique_tracks[:50]  # Limit to top 50 for analysis

    def _generate_search_queries(self, project_context: Dict, genre_recommendations: List[Dict]) -> List[Dict]:
        """Generate targeted search queries for different projects"""

        project = project_context.get('project', 'Generic')
        queries = []

        if project == "Cyberpunk Goblin Slayer":
            # Cyberpunk-specific searches
            queries.extend([
                {
                    "query": "synthwave cyberpunk instrumental",
                    "priority": 1.0,
                    "filters": {"instrumental": True, "genre": "electronic"}
                },
                {
                    "query": "dark electronic cyberpunk",
                    "priority": 0.9,
                    "filters": {"mood": "dark", "genre": "electronic"}
                },
                {
                    "query": "retrowave neon synthpop",
                    "priority": 0.8,
                    "filters": {"decade": "2010s", "genre": "synthpop"}
                },
                {
                    "query": "cyberpunk 2077 soundtrack style",
                    "priority": 0.85,
                    "filters": {"cinematic": True, "futuristic": True}
                },
                {
                    "query": "industrial electronic dark ambient",
                    "priority": 0.75,
                    "filters": {"ambient": True, "industrial": True}
                }
            ])

        elif project == "Tokyo Debt Desire":
            # Urban drama searches
            queries.extend([
                {
                    "query": "urban drama instrumental tokyo",
                    "priority": 1.0,
                    "filters": {"instrumental": True, "genre": "soundtrack"}
                },
                {
                    "query": "modern japanese instrumental tension",
                    "priority": 0.9,
                    "filters": {"japanese": True, "tension": True}
                },
                {
                    "query": "neo soul urban contemporary",
                    "priority": 0.8,
                    "filters": {"genre": "neo_soul", "contemporary": True}
                },
                {
                    "query": "financial district ambient corporate",
                    "priority": 0.7,
                    "filters": {"ambient": True, "corporate": True}
                }
            ])

        else:
            # Generic anime searches
            queries.extend([
                {
                    "query": "anime instrumental soundtrack",
                    "priority": 1.0,
                    "filters": {"anime": True, "instrumental": True}
                },
                {
                    "query": "j-pop instrumental energetic",
                    "priority": 0.8,
                    "filters": {"jpop": True, "energetic": True}
                }
            ])

        # Add genre-specific searches
        for genre_rec in genre_recommendations[:3]:  # Top 3 genres
            genre = genre_rec['genre']
            priority = genre_rec['priority']

            queries.append({
                "query": f"{genre} instrumental",
                "priority": priority * 0.8,
                "filters": {"genre": genre, "instrumental": True}
            })

        return queries

    async def _call_apple_music_search(self, query: str, limit: int = 25,
                                     filters: Optional[Dict] = None) -> Dict:
        """Call Apple Music search API with error handling"""

        try:
            params = {
                "q": query,
                "limit": limit,
                "types": "songs"
            }

            # Add filters if provided
            if filters:
                for key, value in filters.items():
                    if value:
                        params[f"filter_{key}"] = str(value)

            response = self.session.get(
                f"{self.apple_music_base}/api/search",
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "tracks": data.get("results", {}).get("songs", {}).get("data", []),
                    "query": query
                }
            else:
                logger.warning(f"Apple Music API returned {response.status_code} for query: {query}")
                return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error(f"Apple Music API call failed: {e}")
            return {"success": False, "error": str(e)}

    async def _enhance_track_data(self, track: Dict, query_info: Dict) -> Dict:
        """Enhance track data with BPM estimates and additional metadata"""

        attributes = track.get("attributes", {})

        # Extract basic track info
        enhanced_track = {
            "id": track.get("id", ""),
            "title": attributes.get("name", "Unknown"),
            "artist": attributes.get("artistName", "Unknown"),
            "album": attributes.get("albumName", "Unknown"),
            "duration": attributes.get("durationInMillis", 180000) / 1000.0,
            "genre": attributes.get("genreNames", ["Unknown"])[0],
            "release_date": attributes.get("releaseDate", ""),
            "isrc": attributes.get("isrc", ""),
            "explicit": attributes.get("contentRating") == "explicit",
            "preview_url": "",
            "apple_music_url": "",
            "search_query": query_info['query'],
            "search_priority": query_info['priority']
        }

        # Extract preview URL if available
        previews = attributes.get("previews", [])
        if previews:
            enhanced_track["preview_url"] = previews[0].get("url", "")

        # Estimate BPM from genre and metadata
        enhanced_track["estimated_bpm"] = self._estimate_bpm_from_metadata(enhanced_track)

        # Estimate energy level
        enhanced_track["estimated_energy"] = self._estimate_energy_from_metadata(enhanced_track)

        # Extract mood indicators
        enhanced_track["mood_tags"] = self._extract_mood_tags(enhanced_track)

        # Compatibility tags based on search context
        enhanced_track["compatibility_tags"] = self._generate_compatibility_tags(enhanced_track, query_info)

        return enhanced_track

    def _estimate_bpm_from_metadata(self, track: Dict) -> int:
        """Estimate BPM from track metadata and genre"""

        genre = track.get("genre", "").lower()
        title = track.get("title", "").lower()
        artist = track.get("artist", "").lower()

        # Genre-based BPM estimation
        for genre_key, model in self.genre_bpm_models.items():
            if genre_key.replace("_", " ") in genre or genre_key in genre:
                base_bpm = model["base"]
                variance = model["variance"]

                # Add some randomness within the variance
                bpm_adjustment = np.random.normal(0, base_bpm * variance)
                estimated_bpm = int(base_bpm + bpm_adjustment)

                # Keep within genre range
                min_bpm, max_bpm = model["range"]
                return max(min_bpm, min(max_bpm, estimated_bpm))

        # Title/artist-based hints
        speed_indicators = {
            "fast": 1.2, "quick": 1.15, "rapid": 1.25, "slow": 0.8, "calm": 0.7,
            "intense": 1.3, "energy": 1.1, "chill": 0.75, "ambient": 0.6
        }

        bpm_modifier = 1.0
        for indicator, modifier in speed_indicators.items():
            if indicator in title or indicator in artist:
                bpm_modifier *= modifier

        # Default BPM with modifier
        base_bpm = 120
        estimated_bpm = int(base_bpm * bpm_modifier)

        return max(60, min(180, estimated_bpm))

    def _estimate_energy_from_metadata(self, track: Dict) -> float:
        """Estimate energy level from track metadata"""

        genre = track.get("genre", "").lower()
        title = track.get("title", "").lower()
        duration = track.get("duration", 180)

        # Genre-based energy
        genre_energy = {
            "electronic": 0.8, "techno": 0.9, "house": 0.75, "ambient": 0.3,
            "synthwave": 0.7, "cyberpunk": 0.8, "dubstep": 0.95, "trance": 0.8,
            "drum": 0.9, "industrial": 0.85, "pop": 0.6, "rock": 0.7,
            "jazz": 0.5, "classical": 0.4, "soundtrack": 0.6
        }

        energy = 0.6  # Default
        for genre_key, genre_energy_val in genre_energy.items():
            if genre_key in genre:
                energy = genre_energy_val
                break

        # Title-based adjustments
        energy_words = {
            "energy": 0.1, "power": 0.1, "intense": 0.15, "explosive": 0.2,
            "calm": -0.2, "peaceful": -0.15, "soft": -0.1, "quiet": -0.15,
            "aggressive": 0.2, "hard": 0.1, "heavy": 0.1
        }

        for word, adjustment in energy_words.items():
            if word in title:
                energy += adjustment

        # Duration-based adjustment (shorter tracks tend to be higher energy)
        if duration < 120:  # Under 2 minutes
            energy += 0.1
        elif duration > 300:  # Over 5 minutes
            energy -= 0.1

        return max(0.0, min(1.0, energy))

    def _extract_mood_tags(self, track: Dict) -> List[str]:
        """Extract mood tags from track metadata"""

        title = track.get("title", "").lower()
        genre = track.get("genre", "").lower()
        artist = track.get("artist", "").lower()

        mood_tags = []

        # Title-based mood detection
        mood_keywords = {
            "dark": ["dark", "shadow", "noir", "black"],
            "bright": ["bright", "light", "shine", "glow"],
            "futuristic": ["future", "cyber", "neon", "digital", "tech"],
            "atmospheric": ["ambient", "atmosphere", "space", "ethereal"],
            "intense": ["intense", "power", "force", "strong"],
            "calm": ["calm", "peaceful", "serene", "quiet"],
            "energetic": ["energy", "active", "dynamic", "vibrant"],
            "mysterious": ["mystery", "enigma", "hidden", "secret"],
            "dramatic": ["drama", "epic", "grand", "cinematic"]
        }

        for mood, keywords in mood_keywords.items():
            if any(keyword in title or keyword in artist for keyword in keywords):
                mood_tags.append(mood)

        # Genre-based moods
        if "synthwave" in genre or "cyberpunk" in genre:
            mood_tags.extend(["futuristic", "nostalgic"])
        if "ambient" in genre:
            mood_tags.extend(["atmospheric", "calm"])
        if "industrial" in genre:
            mood_tags.extend(["dark", "intense"])

        return list(set(mood_tags))  # Remove duplicates

    def _generate_compatibility_tags(self, track: Dict, query_info: Dict) -> List[str]:
        """Generate compatibility tags based on search context"""

        tags = []

        # Add search priority indicator
        priority = query_info.get('priority', 0.5)
        if priority >= 0.9:
            tags.append("high_priority_match")
        elif priority >= 0.7:
            tags.append("medium_priority_match")

        # Add query-based tags
        query = query_info.get('query', '').lower()
        if "cyberpunk" in query:
            tags.append("cyberpunk_compatible")
        if "instrumental" in query:
            tags.append("instrumental")
        if "urban" in query:
            tags.append("urban_compatible")

        # Add filters as tags
        filters = query_info.get('filters', {})
        for filter_key, filter_value in filters.items():
            if filter_value:
                tags.append(f"filter_{filter_key}")

        return tags

    def _passes_initial_filter(self, track: Dict, bpm_range: tuple, duration: float) -> bool:
        """Initial filtering to remove obviously incompatible tracks"""

        track_bpm = track.get("estimated_bpm", 120)
        track_duration = track.get("duration", 180)

        # BPM filter (allow some tolerance)
        bpm_min, bpm_max = bpm_range
        bpm_tolerance = 20  # ±20 BPM tolerance
        if not (bpm_min - bpm_tolerance <= track_bpm <= bpm_max + bpm_tolerance):
            return False

        # Duration filter (track should be at least 80% of video duration or easily loopable)
        if track_duration < duration * 0.8 and track_duration < 30:  # Too short and not loopable
            return False

        # Content filter
        if track.get("explicit", False):
            return False  # Filter explicit content for anime

        return True

    def _deduplicate_tracks(self, tracks: List[Dict]) -> List[Dict]:
        """Remove duplicate tracks based on title and artist"""

        seen = set()
        unique_tracks = []

        for track in tracks:
            # Create signature from title and artist
            signature = f"{track.get('title', '').lower()}_{track.get('artist', '').lower()}"
            signature = ''.join(c for c in signature if c.isalnum() or c == '_')

            if signature not in seen:
                seen.add(signature)
                unique_tracks.append(track)

        return unique_tracks

    async def _analyze_track_bpm(self, track: Dict, video_analysis: Dict) -> Dict:
        """Detailed BPM analysis for track-video compatibility"""

        track_bpm = track.get("estimated_bpm", 120)
        track_energy = track.get("estimated_energy", 0.6)
        track_duration = track.get("duration", 180)

        # Video characteristics
        video_bpm_range = video_analysis.get('recommended_bpm_range', (120, 140))
        video_duration = video_analysis.get('metadata', {}).get('duration', 10.0)
        video_energy = video_analysis.get('music_characteristics', {}).get('energy_level', 0.7)
        action_intensity = video_analysis.get('visual_analysis', {}).get('action_intensity', 0.7)
        sync_difficulty = video_analysis.get('sync_difficulty', 0.5)

        # BPM compatibility score
        bpm_min, bpm_max = video_bpm_range
        optimal_bpm = (bpm_min + bpm_max) / 2

        bpm_diff = abs(track_bpm - optimal_bpm)
        bpm_tolerance = (bpm_max - bpm_min) / 2 + 10  # Add 10 BPM tolerance
        bpm_score = max(0, 1 - (bpm_diff / bpm_tolerance))

        # Energy compatibility score
        energy_diff = abs(track_energy - video_energy)
        energy_score = max(0, 1 - energy_diff)

        # Duration compatibility score
        if track_duration >= video_duration:
            duration_score = 1.0
        elif track_duration >= video_duration * 0.7:
            duration_score = 0.8  # Good for looping
        elif track_duration >= 30:
            duration_score = 0.6  # Okay for looping
        else:
            duration_score = 0.3  # Difficult to use

        # Action intensity matching
        # High action videos need energetic music
        if action_intensity > 0.7 and track_energy > 0.6:
            action_bonus = 0.2
        elif action_intensity < 0.4 and track_energy < 0.5:
            action_bonus = 0.1
        else:
            action_bonus = 0

        # Sync potential (easier sync = higher score)
        sync_potential_score = 1 - sync_difficulty

        # Mood compatibility
        video_mood = video_analysis.get('music_characteristics', {}).get('mood', 'balanced')
        track_moods = track.get('mood_tags', [])
        mood_score = self._calculate_mood_score(video_mood, track_moods)

        # Overall compatibility
        compatibility_score = (
            bpm_score * 0.25 +
            energy_score * 0.2 +
            duration_score * 0.15 +
            sync_potential_score * 0.15 +
            mood_score * 0.15 +
            action_bonus * 0.1
        )

        # Sync potential calculation
        sync_potential = (
            compatibility_score * 0.7 +
            bpm_score * 0.2 +
            (1 - sync_difficulty) * 0.1
        )

        return {
            "compatibility_score": min(1.0, compatibility_score),
            "sync_potential": min(1.0, sync_potential),
            "bpm_score": bpm_score,
            "energy_score": energy_score,
            "duration_score": duration_score,
            "mood_score": mood_score,
            "action_bonus": action_bonus,
            "analysis_details": {
                "track_bpm": track_bpm,
                "optimal_bpm": optimal_bpm,
                "bpm_difference": bpm_diff,
                "track_energy": track_energy,
                "video_energy": video_energy,
                "energy_difference": energy_diff,
                "duration_ratio": track_duration / video_duration if video_duration > 0 else 0
            },
            "sync_recommendations": self._generate_sync_recommendations(track, video_analysis, compatibility_score)
        }

    def _calculate_mood_score(self, video_mood: str, track_moods: List[str]) -> float:
        """Calculate mood compatibility score"""

        if not track_moods:
            return 0.5  # Neutral if no mood tags

        # Define mood compatibility matrix
        compatibility_matrix = {
            "dark_intense": {"dark": 1.0, "intense": 0.9, "mysterious": 0.8, "dramatic": 0.7},
            "futuristic_energetic": {"futuristic": 1.0, "energetic": 0.9, "bright": 0.7, "intense": 0.6},
            "tense_modern": {"intense": 0.9, "dramatic": 0.8, "dark": 0.7, "energetic": 0.6},
            "atmospheric": {"atmospheric": 1.0, "calm": 0.8, "mysterious": 0.7, "dark": 0.6},
            "balanced": {"energetic": 0.8, "bright": 0.8, "calm": 0.7, "atmospheric": 0.7}
        }

        mood_scores = []
        video_mood_map = compatibility_matrix.get(video_mood, {})

        for track_mood in track_moods:
            score = video_mood_map.get(track_mood, 0.3)  # Default low compatibility
            mood_scores.append(score)

        return max(mood_scores) if mood_scores else 0.5

    def _generate_sync_recommendations(self, track: Dict, video_analysis: Dict, compatibility_score: float) -> Dict:
        """Generate synchronization recommendations for the track"""

        track_bpm = track.get("estimated_bpm", 120)
        track_duration = track.get("duration", 180)
        video_duration = video_analysis.get('metadata', {}).get('duration', 10.0)

        recommendations = {
            "tempo_adjustment": 0,
            "start_offset": 0,
            "fade_in": 1.0,
            "fade_out": 2.0,
            "loop_strategy": "none",
            "volume_strategy": "standard",
            "sync_confidence": compatibility_score
        }

        # Tempo adjustment recommendation
        optimal_bpm_range = video_analysis.get('recommended_bpm_range', (120, 140))
        optimal_bpm = sum(optimal_bpm_range) / 2

        if abs(track_bpm - optimal_bpm) > 10:
            tempo_adjustment = (optimal_bpm - track_bpm) / track_bpm
            # Limit to prevent audio distortion
            recommendations["tempo_adjustment"] = max(-0.1, min(0.1, tempo_adjustment))

        # Start offset for longer tracks
        if track_duration > video_duration * 1.5:
            # Skip intro section
            recommendations["start_offset"] = min(10.0, track_duration * 0.05)

        # Loop strategy for shorter tracks
        if track_duration < video_duration:
            if track_duration > 30:
                recommendations["loop_strategy"] = "seamless_loop"
            else:
                recommendations["loop_strategy"] = "crossfade_loop"

        # Fade recommendations based on video characteristics
        action_intensity = video_analysis.get('visual_analysis', {}).get('action_intensity', 0.7)

        if action_intensity > 0.8:
            # Fast fade for high action
            recommendations["fade_in"] = 0.5
            recommendations["fade_out"] = 1.0
        elif action_intensity < 0.3:
            # Slow fade for calm scenes
            recommendations["fade_in"] = 2.0
            recommendations["fade_out"] = 3.0

        # Volume strategy
        if action_intensity > 0.7:
            recommendations["volume_strategy"] = "dynamic_peaks"
        elif action_intensity < 0.4:
            recommendations["volume_strategy"] = "gentle_curve"

        return recommendations

    def _create_recommendation_strategy(self, video_analysis: Dict, analyzed_tracks: List[Dict]) -> Dict:
        """Create overall recommendation strategy"""

        if not analyzed_tracks:
            return {"strategy": "fallback", "confidence": 0.0}

        best_track = analyzed_tracks[0]
        best_score = best_track.get('sync_potential', 0)

        strategy = {
            "primary_recommendation": best_track['id'],
            "confidence": best_score,
            "alternative_count": len(analyzed_tracks) - 1,
            "strategy_type": "",
            "optimization_focus": [],
            "potential_issues": []
        }

        # Determine strategy type
        if best_score > 0.85:
            strategy["strategy_type"] = "perfect_match"
        elif best_score > 0.7:
            strategy["strategy_type"] = "good_match_minor_adjustments"
        elif best_score > 0.6:
            strategy["strategy_type"] = "acceptable_match_adjustments_needed"
        else:
            strategy["strategy_type"] = "challenging_match_significant_adjustments"

        # Optimization focus
        bpm_scores = [t.get('bpm_analysis', {}).get('bpm_score', 0) for t in analyzed_tracks[:5]]
        energy_scores = [t.get('bpm_analysis', {}).get('energy_score', 0) for t in analyzed_tracks[:5]]

        if np.mean(bpm_scores) < 0.7:
            strategy["optimization_focus"].append("bpm_alignment")
        if np.mean(energy_scores) < 0.7:
            strategy["optimization_focus"].append("energy_matching")

        # Potential issues
        if best_score < 0.7:
            strategy["potential_issues"].append("low_compatibility_score")

        track_durations = [t.get('duration', 180) for t in analyzed_tracks[:3]]
        video_duration = video_analysis.get('metadata', {}).get('duration', 10.0)

        if all(d < video_duration * 0.8 for d in track_durations):
            strategy["potential_issues"].append("short_track_duration")

        return strategy

    async def get_track_audio_features(self, track_id: str) -> Optional[Dict]:
        """Get detailed audio features for a specific track (if available from Apple Music)"""

        try:
            response = self.session.get(f"{self.apple_music_base}/api/tracks/{track_id}/audio-features")

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Audio features not available for track {track_id}")
                return None

        except Exception as e:
            logger.error(f"Error fetching audio features for track {track_id}: {e}")
            return None


def main():
    """Test the Apple Music BPM analyzer"""

    import sys
    import asyncio

    # Load video analysis results
    try:
        with open("/opt/tower-echo-brain/video_analysis_results.json", 'r') as f:
            video_results = json.load(f)
    except FileNotFoundError:
        logger.error("Video analysis results not found. Run scaled_video_analyzer.py first.")
        sys.exit(1)

    analyzer = AppleMusicBPMAnalyzer()

    # Test with the first video
    video_name = list(video_results.keys())[0]
    video_analysis = video_results[video_name]

    if "error" in video_analysis:
        logger.error(f"Video analysis has errors: {video_analysis['error']}")
        sys.exit(1)

    async def test_analysis():
        logger.info(f"Testing Apple Music BPM analysis for: {video_name}")

        result = await analyzer.analyze_video_for_music_sync(video_analysis)

        print(f"\nApple Music BPM Analysis Results for {video_name}:")
        print(f"Compatible tracks found: {len(result['matching_tracks'])}")
        print(f"Search summary: {result['search_summary']}")
        print(f"Strategy: {result['recommendation_strategy']['strategy_type']}")

        if result['matching_tracks']:
            best_track = result['matching_tracks'][0]
            print(f"\nBest match:")
            print(f"  Title: {best_track['title']}")
            print(f"  Artist: {best_track['artist']}")
            print(f"  BPM: {best_track['estimated_bpm']}")
            print(f"  Sync Potential: {best_track['sync_potential']:.2f}")

        # Save results
        output_file = "/opt/tower-echo-brain/apple_music_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Apple Music analysis results saved to: {output_file}")

    # Run the async test
    asyncio.run(test_analysis())


if __name__ == "__main__":
    main()
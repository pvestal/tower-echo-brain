#!/usr/bin/env python3
"""
Complete Music Integration Pipeline Test
Test the full end-to-end music integration system for Patrick's scaled anime videos
"""
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

from scaled_video_analyzer import ScaledVideoAnalyzer
from music_integration_pipeline import MusicIntegrationPipeline
from apple_music_bpm_analyzer import AppleMusicBPMAnalyzer
from audio_video_mixer import AudioVideoMixer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteMusicIntegrationTester:
    """Test suite for the complete music integration system"""

    def __init__(self):
        self.video_analyzer = ScaledVideoAnalyzer()
        self.apple_music_analyzer = AppleMusicBPMAnalyzer()
        self.audio_mixer = AudioVideoMixer()

        self.test_results = {}
        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output/music_integrated/test_results/")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_complete_integration_test(self, video_path: str) -> Dict[str, Any]:
        """Run complete end-to-end integration test"""

        logger.info(f"Starting complete integration test for: {Path(video_path).name}")
        start_time = time.time()

        test_result = {
            "video_path": video_path,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages": {},
            "errors": [],
            "final_outputs": {},
            "performance_metrics": {}
        }

        try:
            # Stage 1: Video Analysis
            logger.info("Stage 1: Analyzing video characteristics...")
            stage1_start = time.time()

            video_analysis = self.video_analyzer.analyze_video_file(video_path)

            test_result["stages"]["video_analysis"] = {
                "status": "completed",
                "duration": time.time() - stage1_start,
                "output": video_analysis
            }

            logger.info(f"Video analysis completed: {video_analysis['project_context']['project']}")

            # Stage 2: Apple Music BPM Analysis
            logger.info("Stage 2: Apple Music BPM analysis...")
            stage2_start = time.time()

            try:
                music_analysis = await self.apple_music_analyzer.analyze_video_for_music_sync(video_analysis)

                test_result["stages"]["apple_music_analysis"] = {
                    "status": "completed",
                    "duration": time.time() - stage2_start,
                    "tracks_found": len(music_analysis.get('matching_tracks', [])),
                    "best_match_score": music_analysis.get('matching_tracks', [{}])[0].get('sync_potential', 0) if music_analysis.get('matching_tracks') else 0
                }

                logger.info(f"Apple Music analysis: {len(music_analysis.get('matching_tracks', []))} tracks found")

            except Exception as e:
                logger.warning(f"Apple Music analysis failed: {e}")
                test_result["stages"]["apple_music_analysis"] = {
                    "status": "failed",
                    "error": str(e),
                    "duration": time.time() - stage2_start
                }
                test_result["errors"].append(f"Apple Music analysis: {e}")

            # Stage 3: Music Selection and Sync Configuration
            logger.info("Stage 3: Music selection and sync configuration...")
            stage3_start = time.time()

            # Select best music track (from Apple Music or fallback)
            best_music = self._select_best_music(video_analysis, test_result.get("stages", {}).get("apple_music_analysis", {}))

            # Create sync configuration
            sync_config = self._create_test_sync_config(video_analysis, best_music)

            test_result["stages"]["music_selection"] = {
                "status": "completed",
                "duration": time.time() - stage3_start,
                "selected_music": best_music,
                "sync_config": sync_config
            }

            logger.info(f"Selected music: {best_music['title']} (BPM: {best_music.get('bpm', 'unknown')})")

            # Stage 4: Audio-Video Mixing
            logger.info("Stage 4: Audio-video mixing...")
            stage4_start = time.time()

            output_name = f"complete_test_{Path(video_path).stem}_{int(time.time())}.mp4"

            mixing_result = self.audio_mixer.mix_video_with_music(
                video_path, best_music, sync_config, output_name
            )

            test_result["stages"]["audio_video_mixing"] = {
                "status": "completed",
                "duration": time.time() - stage4_start,
                "output_file": mixing_result['output_video'],
                "processing_stats": mixing_result['processing_stats']
            }

            test_result["final_outputs"]["video_with_music"] = mixing_result['output_video']

            logger.info(f"Audio-video mixing completed: {mixing_result['output_video']}")

            # Stage 5: Quality Assessment
            logger.info("Stage 5: Quality assessment...")
            stage5_start = time.time()

            quality_assessment = self._assess_output_quality(mixing_result, video_analysis, best_music)

            test_result["stages"]["quality_assessment"] = {
                "status": "completed",
                "duration": time.time() - stage5_start,
                "assessment": quality_assessment
            }

            # Clean up temporary files
            if "temp_files" in mixing_result:
                self.audio_mixer.cleanup_temp_files(mixing_result["temp_files"])

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            test_result["errors"].append(f"Critical error: {e}")

        # Calculate overall performance metrics
        total_duration = time.time() - start_time
        test_result["performance_metrics"] = {
            "total_duration": total_duration,
            "stages_completed": len([s for s in test_result["stages"].values() if s.get("status") == "completed"]),
            "stages_failed": len([s for s in test_result["stages"].values() if s.get("status") == "failed"]),
            "overall_success": len(test_result["errors"]) == 0
        }

        logger.info(f"Complete integration test finished in {total_duration:.2f}s")

        # Save test results
        self._save_test_results(test_result)

        return test_result

    def _select_best_music(self, video_analysis: Dict, apple_music_stage: Dict) -> Dict[str, Any]:
        """Select the best music track for the video"""

        # Try to use Apple Music results first
        if apple_music_stage.get("status") == "completed" and "matching_tracks" in apple_music_stage:
            apple_tracks = apple_music_stage.get("matching_tracks", [])
            if apple_tracks:
                best_apple_track = apple_tracks[0]
                logger.info(f"Using Apple Music track: {best_apple_track.get('title', 'Unknown')}")
                return best_apple_track

        # Fallback to internal music database
        project = video_analysis['project_context']['project']
        bpm_range = video_analysis['recommended_bpm_range']
        energy_level = video_analysis['music_characteristics']['energy_level']

        if project == "Cyberpunk Goblin Slayer":
            return {
                "id": "fallback_cyberpunk_enhanced",
                "title": "Cyberpunk Shadows",
                "artist": "AI Composer",
                "bpm": int((bpm_range[0] + bpm_range[1]) / 2),
                "energy": energy_level,
                "mood": "dark_intense",
                "tags": ["cyberpunk", "synthwave", "electronic"],
                "source": "internal",
                "compatibility_score": 0.85,
                "sync_potential": 0.80
            }
        else:
            return {
                "id": "fallback_general_enhanced",
                "title": "Anime Journey",
                "artist": "AI Composer",
                "bpm": int((bpm_range[0] + bpm_range[1]) / 2),
                "energy": energy_level,
                "mood": "balanced",
                "tags": ["anime", "instrumental"],
                "source": "internal",
                "compatibility_score": 0.75,
                "sync_potential": 0.70
            }

    def _create_test_sync_config(self, video_analysis: Dict, music_info: Dict) -> Dict[str, Any]:
        """Create comprehensive sync configuration for testing"""

        duration = video_analysis['metadata']['duration']
        pacing = video_analysis['pacing_analysis']
        scene_changes = pacing.get('scene_changes', [])
        action_intensity = video_analysis['visual_analysis']['action_intensity']

        # Generate dynamic volume curve based on video characteristics
        volume_curve = self._generate_dynamic_volume_curve(duration, action_intensity, scene_changes)

        # Calculate sync points
        track_bpm = music_info.get('bpm', 120)
        beat_interval = 60 / track_bpm
        sync_points = [round(change / beat_interval) * beat_interval for change in scene_changes]

        # Determine loop strategy
        music_duration = music_info.get('duration', 180)
        loop_config = None
        if music_duration < duration:
            loop_config = {
                "loop_start": 5.0,  # Skip intro
                "loop_end": music_duration - 3.0,  # Before outro
                "loops_needed": int(duration / (music_duration - 8.0)) + 1,
                "crossfade_duration": beat_interval * 0.5
            }

        return {
            "video_duration": duration,
            "music_duration": music_duration,
            "sync_points": sync_points,
            "volume_curve": volume_curve,
            "fade_in": min(1.0, duration * 0.05),
            "fade_out": min(2.0, duration * 0.1),
            "loop_configuration": loop_config,
            "beat_interval": beat_interval,
            "tempo_adjustment": self._calculate_optimal_tempo_adjustment(music_info, video_analysis)
        }

    def _generate_dynamic_volume_curve(self, duration: float, action_intensity: float, scene_changes: List[float]) -> List[Dict]:
        """Generate dynamic volume curve with multiple peaks"""

        curve = [
            {"time": 0, "volume": 0.0},  # Start silent
            {"time": duration * 0.05, "volume": 0.3},  # Fade in
        ]

        # Add volume changes at scene changes
        base_volume = action_intensity * 0.8
        for i, scene_time in enumerate(scene_changes):
            if 0 < scene_time < duration * 0.95:
                # Alternate between high and medium volume
                volume = base_volume if i % 2 == 0 else base_volume * 0.7
                curve.append({"time": scene_time, "volume": volume})

        # Climax at 2/3 point
        climax_time = duration * 0.67
        curve.append({"time": climax_time, "volume": min(1.0, action_intensity * 1.1)})

        # Fade out
        curve.extend([
            {"time": duration * 0.9, "volume": base_volume * 0.5},
            {"time": duration, "volume": 0.0}
        ])

        # Sort by time and remove duplicates
        curve.sort(key=lambda x: x["time"])
        unique_curve = []
        last_time = -1
        for point in curve:
            if point["time"] > last_time + 0.1:  # Minimum 0.1s between points
                unique_curve.append(point)
                last_time = point["time"]

        return unique_curve

    def _calculate_optimal_tempo_adjustment(self, music_info: Dict, video_analysis: Dict) -> float:
        """Calculate optimal tempo adjustment"""

        track_bpm = music_info.get('bpm', 120)
        optimal_bpm_range = video_analysis['recommended_bpm_range']
        optimal_bpm = sum(optimal_bpm_range) / 2

        if abs(track_bpm - optimal_bpm) > 10:
            adjustment = (optimal_bpm - track_bpm) / track_bpm
            return max(-0.1, min(0.1, adjustment))  # Limit to ±10%

        return 0.0

    def _assess_output_quality(self, mixing_result: Dict, video_analysis: Dict, music_info: Dict) -> Dict[str, Any]:
        """Assess the quality of the final output"""

        assessment = {
            "overall_score": 0.0,
            "technical_quality": {},
            "sync_quality": {},
            "aesthetic_quality": {},
            "recommendations": []
        }

        # Technical quality assessment
        processing_stats = mixing_result.get('processing_stats', {})

        assessment["technical_quality"] = {
            "file_integrity": processing_stats.get('validation_passed', False),
            "audio_quality": "good" if processing_stats.get('audio_bitrate', 0) >= 128000 else "low",
            "duration_accuracy": abs(processing_stats.get('duration', 0) - video_analysis['metadata']['duration']) < 0.5,
            "file_size_reasonable": processing_stats.get('file_size', 0) > 0
        }

        technical_score = sum(1 for v in assessment["technical_quality"].values() if v in [True, "good"]) / len(assessment["technical_quality"])

        # Sync quality assessment
        music_bpm = music_info.get('bpm', 120)
        video_bpm_range = video_analysis['recommended_bpm_range']
        bpm_compatibility = video_bpm_range[0] <= music_bpm <= video_bpm_range[1]

        energy_compatibility = abs(music_info.get('energy', 0.5) - video_analysis['music_characteristics']['energy_level']) < 0.3

        assessment["sync_quality"] = {
            "bpm_compatibility": bpm_compatibility,
            "energy_match": energy_compatibility,
            "mood_appropriateness": self._assess_mood_match(music_info, video_analysis),
            "duration_handling": mixing_result.get('sync_config', {}).get('loop_configuration') is not None if music_info.get('duration', 180) < video_analysis['metadata']['duration'] else True
        }

        sync_score = sum(1 for v in assessment["sync_quality"].values() if v is True) / len(assessment["sync_quality"])

        # Aesthetic quality assessment
        project_genre = video_analysis['project_context']['genre']
        music_tags = music_info.get('tags', [])

        genre_match = False
        if project_genre == "cyberpunk_action" and any(tag in music_tags for tag in ["cyberpunk", "synthwave", "electronic"]):
            genre_match = True
        elif project_genre == "urban_drama" and any(tag in music_tags for tag in ["urban", "dramatic", "modern"]):
            genre_match = True

        assessment["aesthetic_quality"] = {
            "genre_appropriateness": genre_match,
            "cultural_fit": "japanese" in video_analysis['project_context'].get('cultural_context', ''),
            "action_intensity_match": abs(music_info.get('energy', 0.5) - video_analysis['visual_analysis']['action_intensity']) < 0.2,
            "professional_quality": technical_score > 0.8
        }

        aesthetic_score = sum(1 for v in assessment["aesthetic_quality"].values() if v is True) / len(assessment["aesthetic_quality"])

        # Overall score
        assessment["overall_score"] = (technical_score * 0.3 + sync_score * 0.4 + aesthetic_score * 0.3)

        # Generate recommendations
        if technical_score < 0.8:
            assessment["recommendations"].append("Improve technical processing pipeline")
        if sync_score < 0.7:
            assessment["recommendations"].append("Better BPM and energy matching needed")
        if aesthetic_score < 0.6:
            assessment["recommendations"].append("Consider genre-specific music selection")
        if assessment["overall_score"] > 0.9:
            assessment["recommendations"].append("Excellent integration - ready for production")

        return assessment

    def _assess_mood_match(self, music_info: Dict, video_analysis: Dict) -> bool:
        """Assess if music mood matches video mood"""

        video_mood = video_analysis['music_characteristics'].get('mood', 'balanced')
        music_mood = music_info.get('mood', 'balanced')
        music_tags = music_info.get('mood_tags', [music_mood])

        # Define mood compatibility
        compatible_moods = {
            "dark_intense": ["dark", "intense", "mysterious"],
            "futuristic_energetic": ["futuristic", "energetic", "bright"],
            "tense_modern": ["tense", "dramatic", "intense"],
            "atmospheric": ["atmospheric", "ambient", "calm"],
            "balanced": ["balanced", "energetic", "dramatic"]
        }

        compatible_list = compatible_moods.get(video_mood, [video_mood])
        return any(mood in music_tags for mood in compatible_list)

    def _save_test_results(self, test_result: Dict[str, Any]):
        """Save comprehensive test results"""

        video_name = Path(test_result["video_path"]).stem
        timestamp = int(time.time())

        # Save detailed results
        results_file = self.output_dir / f"integration_test_{video_name}_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(test_result, f, indent=2, default=str)

        # Save summary report
        summary = {
            "video": video_name,
            "timestamp": test_result["test_timestamp"],
            "success": test_result["performance_metrics"]["overall_success"],
            "total_duration": test_result["performance_metrics"]["total_duration"],
            "stages_completed": test_result["performance_metrics"]["stages_completed"],
            "final_output": test_result["final_outputs"].get("video_with_music", "None"),
            "quality_score": test_result["stages"].get("quality_assessment", {}).get("assessment", {}).get("overall_score", 0),
            "errors": test_result["errors"]
        }

        summary_file = self.output_dir / f"test_summary_{video_name}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Test results saved: {results_file}")
        logger.info(f"Test summary saved: {summary_file}")

    async def run_batch_test(self, video_directory: str) -> Dict[str, Any]:
        """Run integration tests on multiple videos"""

        video_dir = Path(video_directory)
        video_files = list(video_dir.glob("*.mp4"))

        logger.info(f"Running batch test on {len(video_files)} videos from {video_dir}")

        batch_results = {
            "batch_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "video_directory": str(video_dir),
            "total_videos": len(video_files),
            "results": {},
            "summary": {}
        }

        successful_tests = 0
        total_processing_time = 0

        for video_file in video_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {video_file.name}")
            logger.info(f"{'='*60}")

            try:
                result = await self.run_complete_integration_test(str(video_file))
                batch_results["results"][video_file.name] = result

                if result["performance_metrics"]["overall_success"]:
                    successful_tests += 1

                total_processing_time += result["performance_metrics"]["total_duration"]

            except Exception as e:
                logger.error(f"Batch test failed for {video_file.name}: {e}")
                batch_results["results"][video_file.name] = {"error": str(e)}

        # Generate batch summary
        batch_results["summary"] = {
            "successful_tests": successful_tests,
            "failed_tests": len(video_files) - successful_tests,
            "success_rate": successful_tests / len(video_files) if video_files else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(video_files) if video_files else 0
        }

        # Save batch results
        batch_file = self.output_dir / f"batch_test_results_{int(time.time())}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)

        logger.info(f"\nBatch test completed:")
        logger.info(f"Success rate: {batch_results['summary']['success_rate']:.1%}")
        logger.info(f"Total processing time: {total_processing_time:.2f}s")
        logger.info(f"Results saved: {batch_file}")

        return batch_results


async def main():
    """Main test function"""

    tester = CompleteMusicIntegrationTester()

    # Test individual video
    test_video = "/mnt/1TB-storage/ComfyUI/output/rife_scaling_tests/cyberpunk_goblin/cyberpunk_goblin_10sec_rife_00001.mp4"

    if Path(test_video).exists():
        logger.info("Running complete music integration test...")

        result = await tester.run_complete_integration_test(test_video)

        print(f"\n{'='*80}")
        print(f"COMPLETE INTEGRATION TEST RESULTS")
        print(f"{'='*80}")
        print(f"Video: {Path(result['video_path']).name}")
        print(f"Overall Success: {'✅ YES' if result['performance_metrics']['overall_success'] else '❌ NO'}")
        print(f"Total Duration: {result['performance_metrics']['total_duration']:.2f}s")
        print(f"Stages Completed: {result['performance_metrics']['stages_completed']}/5")

        if result['final_outputs'].get('video_with_music'):
            print(f"Final Output: {result['final_outputs']['video_with_music']}")

        if result.get('stages', {}).get('quality_assessment', {}).get('assessment'):
            quality = result['stages']['quality_assessment']['assessment']
            print(f"Quality Score: {quality['overall_score']:.2f}/1.0")

        if result['errors']:
            print(f"Errors: {len(result['errors'])}")
            for error in result['errors']:
                print(f"  - {error}")
        else:
            print("No errors encountered ✅")

        print(f"\nDetailed results saved to: {tester.output_dir}")

        # Optional: Run batch test on all videos
        # batch_result = await tester.run_batch_test("/mnt/1TB-storage/ComfyUI/output/rife_scaling_tests/cyberpunk_goblin/")

    else:
        logger.error(f"Test video not found: {test_video}")


if __name__ == "__main__":
    asyncio.run(main())
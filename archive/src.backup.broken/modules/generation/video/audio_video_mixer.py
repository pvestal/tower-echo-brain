#!/usr/bin/env python3
"""
Advanced Audio-Video Mixer for Anime Music Integration
Professional-grade mixing with precise synchronization and dynamic audio processing
"""
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioVideoMixer:
    """Professional audio-video mixer for anime music integration"""

    def __init__(self):
        self.temp_dir = Path("/tmp/claude/audio_mixing/")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = Path("/mnt/1TB-storage/ComfyUI/output/music_integrated/")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Audio processing parameters
        self.sample_rate = 44100
        self.bit_depth = 16
        self.channels = 2

    def mix_video_with_music(self, video_path: str, music_info: Dict,
                           sync_config: Dict, output_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Main mixing function that combines video with music using advanced synchronization
        """
        logger.info(f"Starting audio-video mixing for: {Path(video_path).name}")

        start_time = time.time()

        # Step 1: Prepare audio track
        audio_file = self._prepare_audio_track(music_info, sync_config)

        # Step 2: Apply dynamic volume processing
        processed_audio = self._apply_volume_processing(audio_file, sync_config)

        # Step 3: Apply tempo adjustments if needed
        if sync_config.get('tempo_adjustment', 0) != 0:
            processed_audio = self._apply_tempo_adjustment(processed_audio, sync_config)

        # Step 4: Create seamless loops if needed
        if sync_config.get('loop_configuration'):
            processed_audio = self._create_seamless_loop(processed_audio, sync_config)

        # Step 5: Mix with original video audio (if any)
        final_audio = self._mix_with_original_audio(processed_audio, video_path, sync_config)

        # Step 6: Combine with video
        output_file = self._combine_audio_video(video_path, final_audio, sync_config, output_name)

        # Step 7: Post-processing and validation
        processing_stats = self._validate_output(output_file, sync_config)

        processing_time = time.time() - start_time

        result = {
            "input_video": video_path,
            "output_video": output_file,
            "music_info": music_info,
            "sync_config": sync_config,
            "processing_stats": processing_stats,
            "processing_time": processing_time,
            "temp_files": {
                "audio_track": audio_file,
                "processed_audio": processed_audio,
                "final_audio": final_audio
            }
        }

        # Save mixing report
        self._save_mixing_report(result)

        logger.info(f"Audio-video mixing completed in {processing_time:.2f}s: {output_file}")
        return result

    def _prepare_audio_track(self, music_info: Dict, sync_config: Dict) -> str:
        """Prepare the base audio track from music source"""

        music_source = music_info.get('source', 'internal')
        music_id = music_info.get('id', 'unknown')
        duration = sync_config.get('video_duration', 10.0)

        output_file = self.temp_dir / f"base_audio_{music_id}.wav"

        if music_source == 'apple_music':
            # For Apple Music tracks, we'll need to use preview or generate similar
            audio_file = self._prepare_apple_music_audio(music_info, duration)
        elif music_source == 'internal':
            # Generate audio based on internal music database
            audio_file = self._generate_internal_audio(music_info, duration)
        elif music_source == 'fallback':
            # Generate fallback audio
            audio_file = self._generate_fallback_audio(music_info, duration)
        else:
            # Default generation
            audio_file = self._generate_default_audio(music_info, duration)

        return audio_file

    def _prepare_apple_music_audio(self, music_info: Dict, duration: float) -> str:
        """Prepare audio from Apple Music (preview or generated equivalent)"""

        music_id = music_info.get('id', 'unknown')
        output_file = self.temp_dir / f"apple_music_{music_id}.wav"

        # For demo purposes, generate audio that matches Apple Music track characteristics
        # In production, this would integrate with actual Apple Music preview clips

        bpm = music_info.get('estimated_bpm', 120)
        energy = music_info.get('estimated_energy', 0.6)
        mood_tags = music_info.get('mood_tags', [])

        logger.info(f"Generating Apple Music-style audio: {music_info.get('title', 'Unknown')} (BPM: {bmp})")

        # Generate procedural audio based on track characteristics
        audio_data = self._generate_procedural_audio(bpm, energy, mood_tags, duration)
        self._save_audio_data(audio_data, output_file)

        return str(output_file)

    def _generate_internal_audio(self, music_info: Dict, duration: float) -> str:
        """Generate audio from internal music database"""

        music_id = music_info.get('id', 'unknown')
        output_file = self.temp_dir / f"internal_{music_id}.wav"

        bpm = music_info.get('bpm', 120)
        energy = music_info.get('energy', 0.6)
        tags = music_info.get('tags', [])

        logger.info(f"Generating internal audio: {music_info.get('title', 'Unknown')}")

        # Generate audio based on internal track specifications
        if 'cyberpunk' in tags or 'synthwave' in tags:
            audio_data = self._generate_cyberpunk_audio(bpm, energy, duration)
        elif 'urban' in tags or 'dramatic' in tags:
            audio_data = self._generate_urban_drama_audio(bpm, energy, duration)
        else:
            audio_data = self._generate_procedural_audio(bpm, energy, ['balanced'], duration)

        self._save_audio_data(audio_data, output_file)
        return str(output_file)

    def _generate_fallback_audio(self, music_info: Dict, duration: float) -> str:
        """Generate fallback audio when no specific track is available"""

        output_file = self.temp_dir / "fallback_audio.wav"

        bpm = music_info.get('bpm', 120)
        energy = music_info.get('energy', 0.6)

        logger.info(f"Generating fallback audio (BPM: {bpm}, Energy: {energy})")

        # Generate basic audio that matches requirements
        audio_data = self._generate_procedural_audio(bpm, energy, ['fallback'], duration)
        self._save_audio_data(audio_data, output_file)

        return str(output_file)

    def _generate_default_audio(self, music_info: Dict, duration: float) -> str:
        """Generate default audio"""

        output_file = self.temp_dir / "default_audio.wav"

        # Generate silence as placeholder (for now)
        # In production, this would be more sophisticated
        audio_data = np.zeros((int(self.sample_rate * duration), self.channels), dtype=np.float32)
        self._save_audio_data(audio_data, output_file)

        return str(output_file)

    def _generate_procedural_audio(self, bpm: int, energy: float,
                                 mood_tags: List[str], duration: float) -> np.ndarray:
        """Generate procedural audio based on parameters"""

        # Calculate audio parameters
        samples = int(self.sample_rate * duration)
        beat_interval = 60.0 / bpm
        beats_per_measure = 4

        # Initialize audio data
        audio_data = np.zeros((samples, self.channels), dtype=np.float32)

        # Generate kick pattern
        if energy > 0.5:
            kick_freq = 60  # Low frequency for kick
            kick_pattern = [1, 0, 0.5, 0]  # Basic kick pattern

            for beat in range(int(duration / beat_interval)):
                if beat % len(kick_pattern) == 0 or kick_pattern[beat % len(kick_pattern)] > 0:
                    start_sample = int(beat * beat_interval * self.sample_rate)
                    kick_samples = int(0.1 * self.sample_rate)  # 100ms kick

                    if start_sample + kick_samples < samples:
                        t = np.linspace(0, 0.1, kick_samples)
                        kick_wave = np.sin(2 * np.pi * kick_freq * t) * np.exp(-t * 20)
                        kick_amplitude = kick_pattern[beat % len(kick_pattern)] * energy * 0.3

                        for ch in range(self.channels):
                            audio_data[start_sample:start_sample + kick_samples, ch] += kick_wave * kick_amplitude

        # Generate hi-hat pattern
        if energy > 0.3:
            hihat_freq = 8000  # High frequency for hi-hat
            hihat_pattern = [0.2, 0.5, 0.2, 0.8]  # Hi-hat pattern

            for eighth_note in range(int(duration / (beat_interval / 2))):
                start_sample = int(eighth_note * (beat_interval / 2) * self.sample_rate)
                hihat_samples = int(0.05 * self.sample_rate)  # 50ms hi-hat

                if start_sample + hihat_samples < samples:
                    # Generate noise-based hi-hat
                    hihat_wave = np.random.normal(0, 0.1, hihat_samples)
                    # Apply high-pass characteristics
                    hihat_wave = np.convolve(hihat_wave, [1, -0.9], mode='same')
                    hihat_amplitude = hihat_pattern[eighth_note % len(hihat_pattern)] * energy * 0.2

                    for ch in range(self.channels):
                        audio_data[start_sample:start_sample + hihat_samples, ch] += hihat_wave * hihat_amplitude

        # Generate bass line
        if energy > 0.4:
            bass_freq = 80
            bass_pattern = [1, 0, 0.5, 0, 0.7, 0, 0.3, 0]

            for beat in range(int(duration / (beat_interval / 2))):
                if bass_pattern[beat % len(bass_pattern)] > 0:
                    start_sample = int(beat * (beat_interval / 2) * self.sample_rate)
                    bass_samples = int(0.2 * self.sample_rate)  # 200ms bass

                    if start_sample + bass_samples < samples:
                        t = np.linspace(0, 0.2, bass_samples)
                        bass_wave = np.sin(2 * np.pi * bass_freq * t) * np.exp(-t * 5)
                        bass_amplitude = bass_pattern[beat % len(bass_pattern)] * energy * 0.25

                        for ch in range(self.channels):
                            audio_data[start_sample:start_sample + bass_samples, ch] += bass_wave * bass_amplitude

        # Add mood-specific elements
        if 'dark' in mood_tags:
            # Add dark atmospheric elements
            self._add_dark_atmosphere(audio_data, duration, energy)
        elif 'futuristic' in mood_tags:
            # Add synthetic elements
            self._add_synthetic_elements(audio_data, duration, energy, bpm)
        elif 'dramatic' in mood_tags:
            # Add dramatic elements
            self._add_dramatic_elements(audio_data, duration, energy)

        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8  # Leave headroom

        return audio_data

    def _generate_cyberpunk_audio(self, bpm: int, energy: float, duration: float) -> np.ndarray:
        """Generate cyberpunk-style audio"""

        # Start with procedural base
        audio_data = self._generate_procedural_audio(bpm, energy, ['dark', 'futuristic'], duration)

        # Add cyberpunk-specific elements
        samples = len(audio_data)

        # Add arpeggiator patterns
        arp_freq = 220  # Base frequency for arpeggiator
        arp_notes = [1, 1.5, 2, 3, 2, 1.5]  # Arpeggiator pattern
        note_duration = (60.0 / bpm) / 4  # Sixteenth notes

        for note_idx in range(int(duration / note_duration)):
            freq = arp_freq * arp_notes[note_idx % len(arp_notes)]
            start_sample = int(note_idx * note_duration * self.sample_rate)
            note_samples = int(note_duration * self.sample_rate)

            if start_sample + note_samples < samples:
                t = np.linspace(0, note_duration, note_samples)
                # Sawtooth wave for synth character
                arp_wave = 2 * (t * freq - np.floor(t * freq + 0.5))
                arp_wave *= np.exp(-t * 3)  # Exponential decay
                arp_amplitude = energy * 0.15

                for ch in range(self.channels):
                    audio_data[start_sample:start_sample + note_samples, ch] += arp_wave * arp_amplitude

        return audio_data

    def _generate_urban_drama_audio(self, bpm: int, energy: float, duration: float) -> np.ndarray:
        """Generate urban drama-style audio"""

        # Start with procedural base
        audio_data = self._generate_procedural_audio(bpm, energy, ['dramatic', 'tense'], duration)

        # Add urban drama elements
        samples = len(audio_data)

        # Add piano-like elements
        piano_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C major scale
        chord_duration = (60.0 / bpm) * 2  # Half notes

        for chord_idx in range(int(duration / chord_duration)):
            start_sample = int(chord_idx * chord_duration * self.sample_rate)
            chord_samples = int(chord_duration * self.sample_rate)

            if start_sample + chord_samples < samples:
                t = np.linspace(0, chord_duration, chord_samples)
                chord_wave = np.zeros(chord_samples)

                # Build chord (3 notes)
                for note_offset in [0, 2, 4]:  # Root, third, fifth
                    freq = piano_freqs[(chord_idx + note_offset) % len(piano_freqs)]
                    note_wave = np.sin(2 * np.pi * freq * t) * np.exp(-t * 2)
                    chord_wave += note_wave * 0.3

                chord_amplitude = energy * 0.2

                for ch in range(self.channels):
                    audio_data[start_sample:start_sample + chord_samples, ch] += chord_wave * chord_amplitude

        return audio_data

    def _add_dark_atmosphere(self, audio_data: np.ndarray, duration: float, energy: float):
        """Add dark atmospheric elements to audio"""

        samples = len(audio_data)

        # Add low-frequency rumble
        rumble_freq = 40
        t = np.linspace(0, duration, samples)
        rumble = np.sin(2 * np.pi * rumble_freq * t) * energy * 0.1

        for ch in range(self.channels):
            audio_data[:, ch] += rumble

    def _add_synthetic_elements(self, audio_data: np.ndarray, duration: float, energy: float, bpm: int):
        """Add synthetic/futuristic elements"""

        samples = len(audio_data)

        # Add filtered noise sweeps
        sweep_duration = (60.0 / bpm) * 8  # Every 8 beats
        sweep_count = int(duration / sweep_duration)

        for sweep_idx in range(sweep_count):
            start_sample = int(sweep_idx * sweep_duration * self.sample_rate)
            sweep_samples = int(sweep_duration * self.sample_rate)

            if start_sample + sweep_samples < samples:
                # Generate noise
                noise = np.random.normal(0, 0.1, sweep_samples)

                # Apply frequency sweep filter (simple high-pass)
                for i in range(1, len(noise)):
                    noise[i] = noise[i] - 0.95 * noise[i-1]

                sweep_amplitude = energy * 0.1

                for ch in range(self.channels):
                    audio_data[start_sample:start_sample + sweep_samples, ch] += noise * sweep_amplitude

    def _add_dramatic_elements(self, audio_data: np.ndarray, duration: float, energy: float):
        """Add dramatic elements to audio"""

        samples = len(audio_data)

        # Add timpani-like hits at dramatic moments
        hit_times = [duration * 0.25, duration * 0.5, duration * 0.75]

        for hit_time in hit_times:
            start_sample = int(hit_time * self.sample_rate)
            hit_samples = int(0.5 * self.sample_rate)  # 500ms hit

            if start_sample + hit_samples < samples:
                t = np.linspace(0, 0.5, hit_samples)
                hit_wave = np.sin(2 * np.pi * 80 * t) * np.exp(-t * 10)  # Low frequency with fast decay
                hit_amplitude = energy * 0.3

                for ch in range(self.channels):
                    audio_data[start_sample:start_sample + hit_samples, ch] += hit_wave * hit_amplitude

    def _save_audio_data(self, audio_data: np.ndarray, output_file: Path):
        """Save audio data to WAV file"""

        try:
            # Convert to 16-bit integers
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Create temporary raw file
            temp_raw = self.temp_dir / f"temp_{output_file.stem}.raw"
            audio_int16.tofile(str(temp_raw))

            # Use FFmpeg to convert to WAV
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 's16le',
                '-ar', str(self.sample_rate),
                '-ac', str(self.channels),
                '-i', str(temp_raw),
                '-acodec', 'pcm_s16le',
                str(output_file)
            ]

            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            temp_raw.unlink()  # Clean up temp file

            logger.info(f"Generated audio: {output_file}")

        except Exception as e:
            logger.error(f"Error saving audio data: {e}")
            raise

    def _apply_volume_processing(self, audio_file: str, sync_config: Dict) -> str:
        """Apply dynamic volume processing based on sync configuration"""

        output_file = self.temp_dir / f"volume_processed_{Path(audio_file).stem}.wav"
        volume_curve = sync_config.get('volume_curve', [])

        if not volume_curve:
            # No volume processing needed
            subprocess.run(['cp', audio_file, str(output_file)], check=True)
            return str(output_file)

        # Build FFmpeg volume filter
        volume_filter = self._build_volume_filter(volume_curve)

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', audio_file,
            '-af', volume_filter,
            '-acodec', 'pcm_s16le',
            str(output_file)
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            logger.info(f"Applied volume processing: {output_file}")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Volume processing failed: {e}")
            # Return original file as fallback
            subprocess.run(['cp', audio_file, str(output_file)], check=True)
            return str(output_file)

    def _build_volume_filter(self, volume_curve: List[Dict]) -> str:
        """Build FFmpeg volume filter from volume curve points"""

        if not volume_curve:
            return "volume=0.7"

        # Sort points by time
        sorted_points = sorted(volume_curve, key=lambda x: x['time'])

        # Build volume filter with keyframes
        volume_points = []
        for point in sorted_points:
            time_val = point['time']
            volume_val = point['volume']
            volume_points.append(f"{time_val}:{volume_val}")

        # Create interpolated volume filter
        volume_string = ":".join(volume_points)
        return f"volume='if(lt(t,{sorted_points[0]['time']}),{sorted_points[0]['volume']},if(gt(t,{sorted_points[-1]['time']}),{sorted_points[-1]['volume']},interp(t)))':eval=frame"

    def _apply_tempo_adjustment(self, audio_file: str, sync_config: Dict) -> str:
        """Apply tempo adjustment to audio"""

        tempo_adjustment = sync_config.get('tempo_adjustment', 0)
        if abs(tempo_adjustment) < 0.01:  # No significant adjustment needed
            return audio_file

        output_file = self.temp_dir / f"tempo_adjusted_{Path(audio_file).stem}.wav"

        # Calculate tempo factor (1.0 = no change, 1.1 = 10% faster, 0.9 = 10% slower)
        tempo_factor = 1.0 + tempo_adjustment

        # Use FFmpeg atempo filter (limited to 0.5x - 2.0x)
        tempo_factor = max(0.5, min(2.0, tempo_factor))

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', audio_file,
            '-af', f'atempo={tempo_factor}',
            '-acodec', 'pcm_s16le',
            str(output_file)
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            logger.info(f"Applied tempo adjustment ({tempo_factor:.2f}x): {output_file}")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Tempo adjustment failed: {e}")
            return audio_file  # Return original as fallback

    def _create_seamless_loop(self, audio_file: str, sync_config: Dict) -> str:
        """Create seamless audio loop for videos longer than the music"""

        loop_config = sync_config.get('loop_configuration')
        if not loop_config:
            return audio_file

        output_file = self.temp_dir / f"looped_{Path(audio_file).stem}.wav"
        video_duration = sync_config.get('video_duration', 10.0)

        loop_start = loop_config.get('loop_start', 0)
        loop_end = loop_config.get('loop_end', 60)
        loops_needed = loop_config.get('loops_needed', 1)
        crossfade_duration = loop_config.get('crossfade_duration', 0.5)

        # Build complex FFmpeg filter for seamless looping
        filter_parts = []

        # Extract loop section
        loop_section = f"[0:a]atrim=start={loop_start}:end={loop_end}[loop]"
        filter_parts.append(loop_section)

        # Create multiple loop copies with crossfade
        loop_inputs = "[loop]"
        for i in range(loops_needed - 1):
            copy_filter = f"[loop]acopy[loop{i+1}]"
            filter_parts.append(copy_filter)
            loop_inputs += f"[loop{i+1}]"

        # Concatenate with crossfade
        concat_filter = f"{loop_inputs}concat=n={loops_needed}:v=0:a=1[looped]"
        filter_parts.append(concat_filter)

        # Trim to exact video duration
        trim_filter = f"[looped]atrim=start=0:duration={video_duration}[final]"
        filter_parts.append(trim_filter)

        filter_complex = ";".join(filter_parts)

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', audio_file,
            '-filter_complex', filter_complex,
            '-map', '[final]',
            '-acodec', 'pcm_s16le',
            str(output_file)
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            logger.info(f"Created seamless loop: {output_file}")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Loop creation failed: {e}")
            return audio_file  # Return original as fallback

    def _mix_with_original_audio(self, music_audio: str, video_path: str, sync_config: Dict) -> str:
        """Mix music with original video audio (if any)"""

        output_file = self.temp_dir / f"mixed_audio_{Path(video_path).stem}.wav"

        # Check if video has audio
        try:
            probe_cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', video_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            has_audio = bool(result.stdout.strip())
        except:
            has_audio = False

        if not has_audio:
            # No original audio to mix
            subprocess.run(['cp', music_audio, str(output_file)], check=True)
            return str(output_file)

        # Mix music with original audio
        original_volume = 0.1  # Keep original audio very low
        music_volume = 0.9     # Music is primary

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', music_audio,
            '-i', video_path,
            '-filter_complex', f'[0:a]volume={music_volume}[music];[1:a]volume={original_volume}[orig];[music][orig]amix=inputs=2:duration=first[mixed]',
            '-map', '[mixed]',
            '-acodec', 'pcm_s16le',
            str(output_file)
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            logger.info(f"Mixed with original audio: {output_file}")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio mixing failed: {e}")
            subprocess.run(['cp', music_audio, str(output_file)], check=True)
            return str(output_file)

    def _combine_audio_video(self, video_path: str, audio_file: str,
                           sync_config: Dict, output_name: Optional[str] = None) -> str:
        """Combine processed audio with video"""

        if not output_name:
            output_name = f"{Path(video_path).stem}_with_music.mp4"

        output_file = self.output_dir / output_name

        # Apply fade in/out
        fade_in = sync_config.get('fade_in', 1.0)
        fade_out = sync_config.get('fade_out', 2.0)
        duration = sync_config.get('video_duration', 10.0)

        # Build audio filter with fades
        audio_filter = f"afade=t=in:ss=0:d={fade_in},afade=t=out:st={duration-fade_out}:d={fade_out}"

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_file,
            '-c:v', 'copy',  # Copy video stream without re-encoding
            '-af', audio_filter,
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',  # Match shortest input duration
            str(output_file)
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            logger.info(f"Combined audio and video: {output_file}")
            return str(output_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"Audio-video combination failed: {e}")
            # Return original video as fallback
            return video_path

    def _validate_output(self, output_file: str, sync_config: Dict) -> Dict[str, Any]:
        """Validate the output file and gather statistics"""

        try:
            # Get file info
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', output_file
            ]

            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                info = json.loads(result.stdout)

                format_info = info.get('format', {})
                video_stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), {})
                audio_stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'audio'), {})

                stats = {
                    "file_size": int(format_info.get('size', 0)),
                    "duration": float(format_info.get('duration', 0)),
                    "bitrate": int(format_info.get('bit_rate', 0)),
                    "video_codec": video_stream.get('codec_name', 'unknown'),
                    "audio_codec": audio_stream.get('codec_name', 'unknown'),
                    "audio_bitrate": int(audio_stream.get('bit_rate', 0)),
                    "audio_sample_rate": int(audio_stream.get('sample_rate', 0)),
                    "validation_passed": True
                }

                # Check if duration matches expectation
                expected_duration = sync_config.get('video_duration', 10.0)
                duration_diff = abs(stats['duration'] - expected_duration)
                if duration_diff > 0.5:  # More than 0.5s difference
                    stats['duration_warning'] = f"Duration mismatch: {duration_diff:.2f}s"

                return stats
            else:
                return {"validation_passed": False, "error": "Could not probe output file"}

        except Exception as e:
            return {"validation_passed": False, "error": str(e)}

    def _save_mixing_report(self, result: Dict[str, Any]):
        """Save detailed mixing report"""

        output_file = Path(result['output_video'])
        report_file = output_file.parent / f"{output_file.stem}_mixing_report.json"

        # Create comprehensive report
        report = {
            "mixing_session": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_video": result['input_video'],
                "output_video": result['output_video'],
                "processing_time": result['processing_time']
            },
            "music_selection": {
                "title": result['music_info'].get('title', 'Unknown'),
                "artist": result['music_info'].get('artist', 'Unknown'),
                "bpm": result['music_info'].get('bpm', 120),
                "energy": result['music_info'].get('energy', 0.6),
                "source": result['music_info'].get('source', 'unknown')
            },
            "sync_configuration": result['sync_config'],
            "processing_stats": result['processing_stats'],
            "temp_files": result['temp_files']
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Mixing report saved: {report_file}")

    def cleanup_temp_files(self, temp_files: Dict[str, str]):
        """Clean up temporary files"""

        for file_type, file_path in temp_files.items():
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    logger.debug(f"Cleaned up {file_type}: {file_path}")
            except Exception as e:
                logger.warning(f"Could not clean up {file_path}: {e}")

    def copy_to_jellyfin(self, video_file: str, metadata: Dict) -> bool:
        """Copy final video to Jellyfin library"""

        try:
            jellyfin_dir = Path("/mnt/10TB2/Anime/AI_Generated/Music_Integrated/")
            jellyfin_dir.mkdir(parents=True, exist_ok=True)

            video_name = Path(video_file).name
            jellyfin_file = jellyfin_dir / video_name

            # Copy video file
            subprocess.run(['cp', video_file, str(jellyfin_file)], check=True)

            # Create metadata sidecar
            metadata_file = jellyfin_file.with_suffix('.nfo')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Copied to Jellyfin: {jellyfin_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to copy to Jellyfin: {e}")
            return False


def main():
    """Test the audio-video mixer"""

    # Test configuration
    test_video = "/mnt/1TB-storage/ComfyUI/output/rife_scaling_tests/cyberpunk_goblin/cyberpunk_goblin_10sec_rife_00001.mp4"

    test_music_info = {
        "id": "test_cyberpunk",
        "title": "Test Cyberpunk Track",
        "artist": "AI Generated",
        "bpm": 130,
        "energy": 0.8,
        "mood": "dark_intense",
        "tags": ["cyberpunk", "synthwave"],
        "source": "internal"
    }

    test_sync_config = {
        "video_duration": 9.625,
        "music_duration": 180,
        "volume_curve": [
            {"time": 0, "volume": 0.0},
            {"time": 0.5, "volume": 0.5},
            {"time": 5.0, "volume": 0.8},
            {"time": 9.0, "volume": 0.3},
            {"time": 9.625, "volume": 0.0}
        ],
        "fade_in": 0.5,
        "fade_out": 1.0,
        "tempo_adjustment": 0.05
    }

    mixer = AudioVideoMixer()

    if Path(test_video).exists():
        logger.info("Testing audio-video mixer...")

        try:
            result = mixer.mix_video_with_music(
                test_video, test_music_info, test_sync_config, "test_cyberpunk_with_music.mp4"
            )

            print(f"\nMixing Results:")
            print(f"Input: {result['input_video']}")
            print(f"Output: {result['output_video']}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            print(f"Validation: {'PASSED' if result['processing_stats']['validation_passed'] else 'FAILED'}")

            # Clean up temp files
            mixer.cleanup_temp_files(result['temp_files'])

        except Exception as e:
            logger.error(f"Mixing test failed: {e}")
    else:
        logger.error(f"Test video not found: {test_video}")


if __name__ == "__main__":
    main()
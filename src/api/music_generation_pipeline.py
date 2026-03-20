"""
Music Generation Pipeline — playlist analysis, ACE-Step generation, scene suggestions, curated playlists.

Endpoints:
  POST /api/music/analyze-playlist           — analyze playlist genres/BPM/mood
  POST /api/music/generate-from-playlist     — generate ACE-Step music inspired by playlist
  GET  /api/music/suggest-for-scene          — Echo Brain AI suggests music params for a scene
  GET  /api/music/curated-playlists          — list curated playlists
  POST /api/music/curated-playlists          — create curated playlist
  DELETE /api/music/curated-playlists/{id}   — delete curated playlist
  POST /api/music/curated-playlists/{id}/tracks   — add track
  DELETE /api/music/curated-playlists/{id}/tracks/{track_id} — remove track
"""

import json
import logging
import os
import urllib.request
from typing import Optional

import asyncpg
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/music", tags=["music_pipeline"])

ACE_STEP_URL = "http://localhost:8440"

# ---------------------------------------------------------------------------
# Genre → BPM/energy/mood mapping
# ---------------------------------------------------------------------------
GENRE_PROFILES = {
    "hip-hop/rap": {"bpm_range": (85, 115), "energy": 0.8, "mood": "dominant"},
    "hip-hop": {"bpm_range": (85, 115), "energy": 0.8, "mood": "dominant"},
    "rap": {"bpm_range": (85, 115), "energy": 0.8, "mood": "dominant"},
    "pop": {"bpm_range": (100, 130), "energy": 0.7, "mood": "energetic"},
    "rock": {"bpm_range": (110, 140), "energy": 0.8, "mood": "powerful"},
    "alternative": {"bpm_range": (100, 135), "energy": 0.7, "mood": "powerful"},
    "electronic": {"bpm_range": (120, 150), "energy": 0.9, "mood": "energetic"},
    "dance": {"bpm_range": (120, 150), "energy": 0.9, "mood": "energetic"},
    "r&b/soul": {"bpm_range": (70, 100), "energy": 0.5, "mood": "romantic"},
    "r&b": {"bpm_range": (70, 100), "energy": 0.5, "mood": "romantic"},
    "soul": {"bpm_range": (70, 100), "energy": 0.5, "mood": "romantic"},
    "jazz": {"bpm_range": (80, 160), "energy": 0.5, "mood": "peaceful"},
    "classical": {"bpm_range": (60, 120), "energy": 0.4, "mood": "peaceful"},
    "soundtrack": {"bpm_range": (70, 140), "energy": 0.6, "mood": "dramatic"},
    "country": {"bpm_range": (90, 130), "energy": 0.6, "mood": "energetic"},
    "metal": {"bpm_range": (120, 180), "energy": 0.95, "mood": "powerful"},
    "reggae": {"bpm_range": (70, 100), "energy": 0.5, "mood": "peaceful"},
    "latin": {"bpm_range": (90, 130), "energy": 0.7, "mood": "energetic"},
    "anime": {"bpm_range": (100, 150), "energy": 0.7, "mood": "dramatic"},
    "j-pop": {"bpm_range": (110, 140), "energy": 0.7, "mood": "energetic"},
}

# Mood → cinematic descriptors
CINEMATIC_MAP = {
    "dominant": "intense urban score, heavy bass, dramatic tension",
    "energetic": "uplifting cinematic orchestral, bright brass, driving percussion",
    "powerful": "epic cinematic score, thunderous drums, heroic brass fanfare",
    "romantic": "tender cinematic strings, gentle piano, emotional warmth",
    "peaceful": "ambient cinematic, gentle harp, atmospheric pads, serene",
    "dramatic": "sweeping film score, full orchestra, emotional crescendo",
    "tense": "dark suspenseful score, low strings, building tension",
    "melancholy": "sorrowful cinematic piano, sparse strings, minor key",
}

ENERGY_DESCRIPTORS = {
    (0.0, 0.3): "calm, gentle, minimal",
    (0.3, 0.5): "moderate, flowing, steady",
    (0.5, 0.7): "upbeat, dynamic, moving",
    (0.7, 0.9): "intense, driving, powerful",
    (0.9, 1.1): "explosive, maximum energy, relentless",
}


def _energy_descriptor(energy: float) -> str:
    for (lo, hi), desc in ENERGY_DESCRIPTORS.items():
        if lo <= energy < hi:
            return desc
    return "moderate"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
async def _get_db_pool():
    """Get or create the database connection pool."""
    if not hasattr(_get_db_pool, "_pool") or _get_db_pool._pool is None:
        _get_db_pool._pool = await asyncpg.create_pool(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            user=os.getenv("DB_USER", "patrick"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "echo_brain"),
            min_size=1,
            max_size=5,
        )
    return _get_db_pool._pool

_get_db_pool._pool = None


async def _db_execute(query: str, *args):
    pool = await _get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("SET search_path = public", )
        return await conn.execute(query, *args)


async def _db_fetch(query: str, *args):
    pool = await _get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("SET search_path = public")
        return await conn.fetch(query, *args)


async def _db_fetchrow(query: str, *args):
    pool = await _get_db_pool()
    async with pool.acquire() as conn:
        await conn.execute("SET search_path = public")
        return await conn.fetchrow(query, *args)


# ---------------------------------------------------------------------------
# ACE-Step helpers
# ---------------------------------------------------------------------------
def _ace_step_request(method: str, path: str, data: dict | None = None) -> dict:
    url = f"{ACE_STEP_URL}{path}"
    if data is not None:
        payload = json.dumps(data).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise HTTPException(status_code=503, detail=f"ACE-Step unavailable: {e}")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PlaylistProfile(BaseModel):
    playlist_id: str
    track_count: int
    avg_bpm: int
    dominant_genres: list[str]
    dominant_mood: str
    energy: float
    genre_breakdown: dict[str, int]


class GenerateFromPlaylistRequest(BaseModel):
    playlist_id: str
    mode: str = "style_matched"  # style_matched | cinematic | vocal_beats
    duration: float = 60.0
    seed: Optional[int] = None


class SuggestResponse(BaseModel):
    suggested_mood: str
    suggested_genre: str
    suggested_bpm: int
    suggested_duration: float
    reasoning: str


class CuratedPlaylistCreate(BaseModel):
    name: str
    description: Optional[str] = None


class CuratedTrackAdd(BaseModel):
    track_id: str
    name: str
    artist: str
    preview_url: str
    source: str  # "ace-step" or "apple-music"


# ---------------------------------------------------------------------------
# Playlist Analysis
# ---------------------------------------------------------------------------
@router.post("/analyze-playlist", response_model=PlaylistProfile)
async def analyze_playlist(body: dict):
    """Analyze an Apple Music playlist — genres, BPM, mood, energy."""
    playlist_id = body.get("playlist_id")
    if not playlist_id:
        raise HTTPException(status_code=400, detail="playlist_id required")

    try:
        from src.integrations.tower_auth_bridge import tower_auth
        result = await tower_auth.get_apple_music_playlist_tracks(playlist_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Apple Music error: {e}")

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    tracks = result.get("tracks", [])
    if not tracks:
        raise HTTPException(status_code=404, detail="Playlist has no tracks")

    # Tally genres from track metadata
    genre_counts: dict[str, int] = {}
    for t in tracks:
        for g in t.get("genres", []):
            key = g.lower().strip()
            genre_counts[key] = genre_counts.get(key, 0) + 1
        # Fallback: use genre_names if present
        for g in t.get("genre_names", []):
            key = g.lower().strip()
            genre_counts[key] = genre_counts.get(key, 0) + 1

    # If no genres from metadata, try to infer from playlist name
    if not genre_counts:
        genre_counts["soundtrack"] = len(tracks)

    # Find dominant genres (top 3)
    sorted_genres = sorted(genre_counts.items(), key=lambda x: -x[1])
    dominant = [g for g, _ in sorted_genres[:3]]

    # Average BPM and energy from genre profiles
    total_bpm = 0
    total_energy = 0.0
    mood_votes: dict[str, int] = {}
    matched = 0

    for genre_name in dominant:
        profile = None
        for key, prof in GENRE_PROFILES.items():
            if key in genre_name or genre_name in key:
                profile = prof
                break
        if profile is None:
            profile = GENRE_PROFILES.get("soundtrack")

        low, high = profile["bpm_range"]
        total_bpm += (low + high) // 2
        total_energy += profile["energy"]
        m = profile["mood"]
        mood_votes[m] = mood_votes.get(m, 0) + 1
        matched += 1

    avg_bpm = total_bpm // max(matched, 1)
    avg_energy = round(total_energy / max(matched, 1), 2)
    dominant_mood = max(mood_votes, key=mood_votes.get) if mood_votes else "dramatic"

    return PlaylistProfile(
        playlist_id=playlist_id,
        track_count=len(tracks),
        avg_bpm=avg_bpm,
        dominant_genres=dominant,
        dominant_mood=dominant_mood,
        energy=avg_energy,
        genre_breakdown=genre_counts,
    )


# ---------------------------------------------------------------------------
# Generate from Playlist
# ---------------------------------------------------------------------------
@router.post("/generate-from-playlist")
async def generate_from_playlist(req: GenerateFromPlaylistRequest):
    """Analyze a playlist and generate music via ACE-Step in the style."""
    # Step 1: analyze
    profile = await analyze_playlist({"playlist_id": req.playlist_id})

    genres_str = ", ".join(profile.dominant_genres[:2])
    energy_desc = _energy_descriptor(profile.energy)

    # Step 2: build prompt based on mode
    if req.mode == "cinematic":
        cinematic_desc = CINEMATIC_MAP.get(profile.dominant_mood, CINEMATIC_MAP["dramatic"])
        prompt = f"cinematic film soundtrack, {cinematic_desc}, {profile.avg_bpm} bpm"
    elif req.mode == "vocal_beats":
        prompt = f"{genres_str} style, {profile.dominant_mood}, {profile.avg_bpm} bpm, {energy_desc}"
        # Generate lyrics via Ollama
        try:
            lyrics = await _generate_lyrics(genres_str, profile.dominant_mood)
        except Exception:
            lyrics = ""
    else:
        # style_matched (default)
        prompt = f"{genres_str} style, {profile.dominant_mood}, {profile.avg_bpm} bpm, {energy_desc}"

    instrumental = req.mode != "vocal_beats"
    lyrics_text = lyrics if req.mode == "vocal_beats" and "lyrics" in dir() else ""

    # Step 3: submit to ACE-Step
    ace_payload = {
        "prompt": prompt,
        "lyrics": lyrics_text,
        "duration": req.duration,
        "format": "wav",
        "instrumental": instrumental,
        "infer_steps": 60,
        "guidance_scale": 15.0,
    }
    if req.seed is not None:
        ace_payload["seed"] = req.seed

    result = _ace_step_request("POST", "/generate", ace_payload)

    return {
        "task_id": result.get("task_id"),
        "status": result.get("status", "pending"),
        "profile": profile.model_dump(),
        "prompt": prompt,
        "mode": req.mode,
        "lyrics": lyrics_text if not instrumental else None,
    }


async def _generate_lyrics(genres: str, mood: str) -> str:
    """Use Ollama to generate short song lyrics."""
    import httpx
    prompt = (
        f"Write short song lyrics (8-12 lines) for a {genres} song with a {mood} mood. "
        f"Just the lyrics, no titles or annotations."
    )
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral:7b", "prompt": prompt, "stream": False,
                      "keep_alive": "5m", "options": {"num_gpu": 0}},
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"Lyrics generation failed: {e}")
    return ""


# ---------------------------------------------------------------------------
# Scene-Based Suggestions (Echo Brain AI)
# ---------------------------------------------------------------------------
@router.get("/suggest-for-scene", response_model=SuggestResponse)
async def suggest_for_scene(
    mood: str = Query(..., description="Scene mood"),
    description: str = Query("", description="Scene description"),
    time_of_day: str = Query("", description="Time of day"),
):
    """Use Ollama to suggest music parameters for a scene."""
    import httpx

    prompt = (
        f"Given an anime scene that is {mood}"
        f"{' at ' + time_of_day if time_of_day else ''}"
        f": {description}\n\n"
        f"Suggest music parameters. Respond in EXACTLY this JSON format, no other text:\n"
        f'{{"mood": "...", "genre": "...", "bpm": 120, "duration": 30, "reasoning": "..."}}'
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral:7b", "prompt": prompt, "stream": False,
                      "keep_alive": "5m", "options": {"num_gpu": 0}},
            )
            if resp.status_code == 200:
                raw = resp.json().get("response", "")
                # Extract JSON from response
                import re
                match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    return SuggestResponse(
                        suggested_mood=data.get("mood", mood),
                        suggested_genre=data.get("genre", "orchestral"),
                        suggested_bpm=int(data.get("bpm", 120)),
                        suggested_duration=float(data.get("duration", 30)),
                        reasoning=data.get("reasoning", "Based on scene mood analysis."),
                    )
    except Exception as e:
        logger.warning(f"Scene suggestion via Ollama failed: {e}")

    # Fallback: rule-based
    fallback_map = {
        "tense": ("tense", "orchestral", 90, 30),
        "romantic": ("romantic", "piano", 80, 45),
        "action": ("action", "electronic", 140, 30),
        "melancholy": ("melancholy", "strings", 75, 45),
        "peaceful": ("peaceful", "ambient", 70, 60),
        "comedic": ("comedic", "woodwinds", 120, 20),
    }
    fb = fallback_map.get(mood.lower(), ("dramatic", "orchestral", 100, 30))
    return SuggestResponse(
        suggested_mood=fb[0],
        suggested_genre=fb[1],
        suggested_bpm=fb[2],
        suggested_duration=fb[3],
        reasoning=f"Rule-based suggestion for '{mood}' scene.",
    )


# ---------------------------------------------------------------------------
# Curated Playlists CRUD
# ---------------------------------------------------------------------------
@router.get("/curated-playlists")
async def list_curated_playlists():
    """List all curated playlists with track counts."""
    rows = await _db_fetch("""
        SELECT p.id, p.name, p.description, p.created_at,
               COUNT(t.id) as track_count
        FROM public.curated_playlists p
        LEFT JOIN public.curated_playlist_tracks t ON t.playlist_id = p.id
        GROUP BY p.id
        ORDER BY p.created_at DESC
    """)
    return {
        "playlists": [
            {
                "id": r["id"],
                "name": r["name"],
                "description": r["description"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "track_count": r["track_count"],
            }
            for r in rows
        ]
    }


@router.post("/curated-playlists")
async def create_curated_playlist(body: CuratedPlaylistCreate):
    """Create a new curated playlist."""
    row = await _db_fetchrow(
        "INSERT INTO public.curated_playlists (name, description) VALUES ($1, $2) RETURNING id, created_at",
        body.name, body.description,
    )
    return {
        "id": row["id"],
        "name": body.name,
        "description": body.description,
        "created_at": row["created_at"].isoformat(),
    }


@router.delete("/curated-playlists/{playlist_id}")
async def delete_curated_playlist(playlist_id: int):
    """Delete a curated playlist and all its tracks."""
    result = await _db_execute(
        "DELETE FROM public.curated_playlists WHERE id = $1", playlist_id
    )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Playlist not found")
    return {"message": "Playlist deleted", "id": playlist_id}


@router.get("/curated-playlists/{playlist_id}/tracks")
async def get_curated_playlist_tracks(playlist_id: int):
    """Get tracks in a curated playlist."""
    rows = await _db_fetch(
        """SELECT id, track_id, track_name, track_artist, preview_url, source, position, added_at
           FROM public.curated_playlist_tracks
           WHERE playlist_id = $1
           ORDER BY position, added_at""",
        playlist_id,
    )
    return {
        "playlist_id": playlist_id,
        "tracks": [
            {
                "id": r["id"],
                "track_id": r["track_id"],
                "track_name": r["track_name"],
                "track_artist": r["track_artist"],
                "preview_url": r["preview_url"],
                "source": r["source"],
                "position": r["position"],
                "added_at": r["added_at"].isoformat() if r["added_at"] else None,
            }
            for r in rows
        ],
    }


@router.post("/curated-playlists/{playlist_id}/tracks")
async def add_track_to_curated_playlist(playlist_id: int, body: CuratedTrackAdd):
    """Add a track to a curated playlist."""
    # Verify playlist exists
    pl = await _db_fetchrow("SELECT id FROM public.curated_playlists WHERE id = $1", playlist_id)
    if not pl:
        raise HTTPException(status_code=404, detail="Playlist not found")

    # Get next position
    max_pos = await _db_fetchrow(
        "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM public.curated_playlist_tracks WHERE playlist_id = $1",
        playlist_id,
    )
    pos = max_pos["next_pos"] if max_pos else 0

    row = await _db_fetchrow(
        """INSERT INTO public.curated_playlist_tracks
           (playlist_id, track_id, track_name, track_artist, preview_url, source, position)
           VALUES ($1, $2, $3, $4, $5, $6, $7)
           RETURNING id""",
        playlist_id, body.track_id, body.name, body.artist, body.preview_url, body.source, pos,
    )
    return {"id": row["id"], "playlist_id": playlist_id, "position": pos}


@router.delete("/curated-playlists/{playlist_id}/tracks/{track_id}")
async def remove_track_from_curated_playlist(playlist_id: int, track_id: str):
    """Remove a track from a curated playlist by track_id string."""
    result = await _db_execute(
        "DELETE FROM public.curated_playlist_tracks WHERE playlist_id = $1 AND track_id = $2",
        playlist_id, track_id,
    )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Track not found in playlist")
    return {"message": "Track removed", "playlist_id": playlist_id, "track_id": track_id}

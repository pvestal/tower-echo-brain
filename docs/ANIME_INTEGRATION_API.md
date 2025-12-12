# Echo Brain Anime Integration API Documentation

## Base URL
`http://localhost:8309/api/echo/anime`

## Authentication
Currently no authentication required (will be added via JWT in future)

## Endpoints

### 1. List All Characters
**GET** `/characters`

Returns all characters in the library with their statistics.

**Response:**
```json
{
  "characters": [
    {
      "id": 1,
      "name": "Yuki",
      "project": "tokyo_debt_desire",
      "reference_image": "/path/to/image.png",
      "stats": {
        "generations": 10,
        "approved": 7,
        "avg_consistency": 0.85
      }
    }
  ]
}
```

### 2. Create Character Profile
**POST** `/character/create`

Create a new character with a reference image.

**Request Body (form-urlencoded):**
- `name` (string, required): Character name
- `project` (string, optional): Project name (default: "tokyo_debt_desire")
- `reference_image` (string, required): Path to reference image

**Response:**
```json
{
  "success": true,
  "character": {
    "id": 5,
    "name": "NewCharacter",
    "project": "tokyo_debt_desire",
    "reference_image": "/path/to/ref.png"
  }
}
```

### 3. Generate Character Variation
**POST** `/character/{character_id}/generate`

Generate a new variation of a character using img2img.

**Path Parameters:**
- `character_id` (int): ID of the character

**Request Body (form-urlencoded):**
- `prompt` (string, required): Description of the desired variation
- `denoise` (float, optional): Denoise strength 0.0-1.0 (default: 0.5)
- `seed` (int, optional): Random seed for reproducibility

**Response:**
```json
{
  "success": true,
  "prompt_id": "uuid-string",
  "generation_id": 42,
  "character": "Yuki",
  "status": "queued"
}
```

### 4. Check Generation Status
**GET** `/generation/{prompt_id}/status`

Check the status of a ComfyUI generation job.

**Path Parameters:**
- `prompt_id` (string): ComfyUI prompt UUID

**Response (Pending):**
```json
{
  "status": "pending"
}
```

**Response (Processing):**
```json
{
  "status": "processing"
}
```

**Response (Completed):**
```json
{
  "status": "completed",
  "image_path": "/mnt/1TB-storage/ComfyUI/output/filename.png",
  "filename": "filename.png"
}
```

**Response (Failed):**
```json
{
  "status": "failed",
  "error": "Error message"
}
```

### 5. Approve Generation
**POST** `/character/{character_id}/approve`

Mark a generation as approved and learn from it.

**Path Parameters:**
- `character_id` (int): ID of the character

**Query Parameters:**
- `generation_id` (int, required): ID of the generation to approve

**Request Body (form-urlencoded):**
- `feedback` (string, optional): User feedback about the generation

**Response:**
```json
{
  "success": true,
  "message": "Generation approved and patterns learned"
}
```

### 6. Reject Generation
**POST** `/character/{character_id}/reject`

Mark a generation as rejected with a reason.

**Path Parameters:**
- `character_id` (int): ID of the character

**Query Parameters:**
- `generation_id` (int, required): ID of the generation to reject

**Request Body (form-urlencoded):**
- `reason` (string, required): Reason for rejection

**Response:**
```json
{
  "success": true,
  "message": "Generation rejected, learning from feedback"
}
```

### 7. Get Character Patterns
**GET** `/character/{character_id}/patterns`

Get learned prompt patterns for a character.

**Path Parameters:**
- `character_id` (int): ID of the character

**Response:**
```json
{
  "character_id": 1,
  "patterns": [
    {
      "type": "pose",
      "prompt": "sitting at dinner table",
      "success_rate": 0.9,
      "usage": 10,
      "approved": 9
    }
  ]
}
```

### 8. Get Character Report
**GET** `/character/{character_id}/report`

Get detailed performance report for a character.

**Path Parameters:**
- `character_id` (int): ID of the character

**Response:**
```json
{
  "character": {
    "id": 1,
    "name": "Yuki",
    "project": "tokyo_debt_desire",
    "reference": "/path/to/ref.png"
  },
  "stats": {
    "total_generations": 50,
    "approved": 35,
    "approval_rate": 0.7,
    "avg_consistency": 0.82
  },
  "recent_generations": [
    {
      "prompt": "sitting elegantly",
      "approved": true,
      "consistency": 0.89,
      "date": "2025-12-12T02:18:01.078579"
    }
  ],
  "best_patterns": [
    {
      "type": "pose",
      "prompt": "sitting at table",
      "success_rate": 0.95
    }
  ]
}
```

### 9. Test ComfyUI Connection
**POST** `/workflow/test`

Test the connection to ComfyUI.

**Response:**
```json
{
  "connected": true,
  "comfyui_version": "3.11",
  "vram": 12288
}
```

### 10. Health Check
**GET** `/health`

Check anime integration service health.

**Response:**
```json
{
  "status": "ok",
  "database": "connected",
  "characters": 4,
  "comfyui": "connected"
}
```

## Error Responses

All endpoints may return error responses:

```json
{
  "detail": "Error message"
}
```

Or for validation errors:

```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "Field required",
      "type": "missing"
    }
  ]
}
```

## Example Workflow

```bash
# 1. Create a character
curl -X POST http://localhost:8309/api/echo/anime/character/create \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "name=Yuki&reference_image=/path/to/yuki.png"

# 2. Generate a variation
curl -X POST http://localhost:8309/api/echo/anime/character/1/generate \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "prompt=sitting at dinner table wearing red dress&denoise=0.5"

# 3. Check status
curl http://localhost:8309/api/echo/anime/generation/abc-123/status

# 4. Approve if good
curl -X POST "http://localhost:8309/api/echo/anime/character/1/approve?generation_id=1" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "feedback=Excellent consistency"

# 5. Check performance
curl http://localhost:8309/api/echo/anime/character/1/report
```

## Notes

- All image paths are absolute paths on the Tower server filesystem
- ComfyUI must be running on port 8188 for generation to work
- Database uses PostgreSQL with pgvector extension for future embedding support
- Generation times vary from 30s to 2min depending on settings
- Denoise values: 0.3-0.4 (very similar), 0.5 (moderate change), 0.6+ (significant variation)
class TowerOrchestrator:
    """REAL Integration with Tower Services for multimedia orchestration"""

    def __init__(self):
        self.services = {
            'comfyui': 'http://127.0.0.1:8188',
            'anime': 'http://127.0.0.1:8328',  # Updated port based on netstat
            'voice': 'http://127.0.0.1:8312',
            'music': 'http://127.0.0.1:8308',
            'kb': 'http://127.0.0.1:8307',
            'ollama': 'http://127.0.0.1:11434'
        }
        self.timeout = 30

    async def generate_image(self, prompt: str, style: str = "anime") -> dict:
        """Generate image using ComfyUI with working workflow"""
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "anything-v4.5.safetensors"}
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": f"{prompt}, {style} style, masterpiece, best quality",
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "worst quality, low quality, blurry",
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1}
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "steps": 20,
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": f"echo_gen_{int(time.time())}",
                    "images": ["6", 0]
                }
            }
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.services['comfyui']}/prompt",
                    json={"prompt": workflow},
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "prompt_id": result.get('prompt_id'),
                            "message": "Image generation started"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"ComfyUI returned status {response.status}"
                        }
        except Exception as e:
            logger.error(f"ComfyUI generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def generate_voice(self, text: str, character: str = "echo_default") -> dict:
        """Generate voice using real Tower voice service"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get auth token first
                token_response = await session.post(
                    f"{self.services['voice']}/api/auth/token",
                    json={"username": "echo_brain", "password": "test"}
                )

                if token_response.status != 200:
                    return {"success": False, "error": "Failed to authenticate with voice service"}

                token_data = await token_response.json()
                token = token_data.get("access_token")

                # Now make the voice request with auth
                async with session.post(
                    f"{self.services['voice']}/api/tts",
                    json={
                        "text": text,
                        "voice": character,
                        "speed": 1.0
                    },
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        return {
                            "success": True,
                            "message": "Voice generated successfully",
                            "character": character,
                            "audio_data": "Binary audio data (not shown)"
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Voice service returned {response.status}: {error_text}"
                        }
        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def create_music(self, description: str, duration: int = 30) -> dict:
        """Create music using Tower music service with proper error handling"""
        # Map descriptions to valid emotional states
        emotion_mapping = {
            "epic": "epic",
            "happy": "energetic",
            "sad": "melancholic",
            "peaceful": "peaceful",
            "action": "heroic",
            "mysterious": "mysterious",
            "romantic": "romantic",
            "funny": "comedic",
            "tense": "tense",
            "thoughtful": "contemplative"
        }

        # Extract emotion from description or use default
        emotion = "epic"
        for keyword, mapped_emotion in emotion_mapping.items():
            if keyword.lower() in description.lower():
                emotion = mapped_emotion
                break

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "style": "electronic",
                    "emotion": emotion,
                    "duration": duration,
                    "bpm": 120,
                    "key_signature": "C major"
                }

                logger.info(f"Creating music with payload: {payload}")

                async with session.post(
                    f"{self.services['music']}/api/generate/music",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response_text = await response.text()

                    if response.status == 200:
                        result = json.loads(response_text)
                        return {
                            "success": True,
                            "message": "Music generation started",
                            "track_id": result.get("track_id"),
                            "title": result.get("title"),
                            "audio_file_path": result.get("audio_file_path"),
                            "duration": result.get("duration"),
                            "bpm": result.get("bpm"),
                            "estimated_time": "30s"
                        }
                    elif response.status == 500:
                        # Parse error message
                        logger.error(f"Music service error: {response_text}")
                        return {
                            "success": False,
                            "error": f"Music service error: {response_text[:200]}"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Music service returned {response.status}: {response_text[:200]}"
                        }
        except asyncio.TimeoutError:
            logger.error("Music generation timed out after 60s")
            return {
                "success": False,
                "error": "Music generation timed out - service may be overloaded"
            }
        except Exception as e:
            logger.error(f"Music generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def orchestrate_multimedia(self, task_type: str, description: str, requirements: dict = {}) -> dict:
        """Orchestrate multimedia tasks across Tower services"""
        logger.info(f"🎬 Orchestrating {task_type}: {description}")

        if task_type == "image":
            prompt = requirements.get("prompt", description)
            style = requirements.get("style", "anime")
            return await self.generate_image(prompt, style)

        elif task_type == "voice":
            text = requirements.get("text", description)
            character = requirements.get("character", "echo_default")
            return await self.generate_voice(text, character)

        elif task_type == "music":
            duration = requirements.get("duration", 30)
            return await self.create_music(description, duration)

        elif task_type == "trailer":
            # Multi-step orchestration
            results = {"steps": []}

            # Generate title card
            title_result = await self.generate_image(
                f"Title card for {description}",
                requirements.get("style", "anime")
            )
            results["steps"].append({"step": "title_card", "result": title_result})

            # Generate voice narration
            narration = requirements.get("narration", f"Coming soon: {description}")
            voice_result = await self.generate_voice(narration, "echo_default")
            results["steps"].append({"step": "narration", "result": voice_result})

            # Generate background music
            music_result = await self.create_music(f"Epic trailer music for {description}", 15)
            results["steps"].append({"step": "music", "result": music_result})

            return {
                "success": True,
                "message": f"Trailer orchestration completed for {description}",
                "orchestration_details": results
            }

        else:
            return {
                "success": False,
                "error": f"Unknown task type: {task_type}"
            }

    async def get_service_status(self, service: str) -> dict:
        """Get status of a specific Tower service"""
        if service not in self.services:
            return {"success": False, "error": f"Unknown service: {service}"}

        try:
            async with aiohttp.ClientSession() as session:
                # Try common health endpoints
                health_endpoints = ["/api/health", "/health", "/system_stats"]

                for endpoint in health_endpoints:
                    try:
                        async with session.get(
                            f"{self.services[service]}{endpoint}",
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                return {
                                    "success": True,
                                    "service": service,
                                    "status": "healthy",
                                    "details": result
                                }
                    except:
                        continue

                return {
                    "success": False,
                    "error": f"No health endpoint found for {service}"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def format_response_for_web(response: str) -> str:
    """Format response text for web display"""
    # Add basic HTML formatting
    response = response.replace('\n', '<br>')
    response = response.replace('```', '<pre>')
    response = response.replace('**', '<strong>')
    return response


def extract_code_blocks(text: str) -> List[Dict]:
    """Extract code blocks from text"""
    code_blocks = []
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        language = match[0] if match[0] else 'text'
        code = match[1].strip()
        code_blocks.append({
            'language': language,
            'code': code
        })

    return code_blocks


def validate_json_structure(data: dict, required_fields: List[str]) -> Tuple[bool, str]:
    """Validate JSON structure has required fields"""
    missing_fields = []
    for field in required_fields:
        if field not in data:
            missing_fields.append(field)

    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    return True, "Valid JSON structure"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')

    # Remove leading/trailing whitespace and dots
    filename = filename.strip('. ')

    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    return filename


# Global utility instances
safe_executor = SafeShellExecutor()
tower_orchestrator = TowerOrchestrator()
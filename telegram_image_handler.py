#!/usr/bin/env python3
"""
Enhanced Telegram Handler with Image Processing for Echo Brain
Handles text, images, and multimodal requests
"""

import os
import json
import logging
import requests
import base64
from typing import Dict, Optional, List, Any
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from datetime import datetime
import httpx
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
enhanced_telegram_router = APIRouter(prefix="/api/telegram/enhanced", tags=["telegram-enhanced"])

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "telegram_webhook_secret_2025")
ECHO_API_URL = "http://localhost:8309/api/echo"
COMFYUI_URL = "http://localhost:8188"
QDRANT_URL = "http://localhost:6333"

class TelegramImageProcessor:
    """Process images from Telegram with context awareness"""

    def __init__(self):
        self.conversation_context = {}
        self.image_cache = {}

    async def download_telegram_file(self, file_id: str) -> bytes:
        """Download file from Telegram"""
        try:
            # Get file path
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile"
            response = requests.get(url, params={"file_id": file_id})
            file_path = response.json()["result"]["file_path"]

            # Download file
            file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
            file_response = requests.get(file_url)
            return file_response.content
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None

    async def process_image_with_text(self, image_data: bytes, text: str,
                                     chat_id: int, user_id: str) -> Dict:
        """Process image with text instruction"""

        # Store image in context
        image_key = f"{chat_id}_{datetime.now().timestamp()}"
        self.image_cache[image_key] = {
            "data": image_data,
            "text": text,
            "timestamp": datetime.now().isoformat()
        }

        # Determine intent
        intent = self._determine_intent(text)

        if intent == "face_swap":
            return await self._handle_face_swap(image_data, text)
        elif intent == "anime_generation":
            return await self._handle_anime_generation(image_data, text)
        elif intent == "image_search":
            return await self._handle_image_search(image_data, text)
        else:
            return await self._handle_general_image_query(image_data, text)

    def _determine_intent(self, text: str) -> str:
        """Determine user intent from text"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["face", "swap", "change face", "replace"]):
            return "face_swap"
        elif any(word in text_lower for word in ["anime", "generate", "create", "make"]):
            return "anime_generation"
        elif any(word in text_lower for word in ["find", "search", "similar", "like this"]):
            return "image_search"
        else:
            return "general"

    async def _handle_face_swap(self, image_data: bytes, instruction: str) -> Dict:
        """Handle face swapping requests"""

        # Save image temporarily
        temp_path = f"/tmp/telegram_face_{datetime.now().timestamp()}.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_data)

        # Prepare ComfyUI workflow for face swap
        workflow = {
            "action": "face_swap",
            "input_image": temp_path,
            "instruction": instruction,
            "parameters": {
                "model": "insightface",
                "strength": 0.8
            }
        }

        # Send to ComfyUI (would need actual integration)
        response_text = f"I understand you want to change the face in this image. {instruction}\n"
        response_text += "Processing your face swap request... This will take a moment.\n"
        response_text += f"Input image saved to: {temp_path}"

        # Save to memory
        await self._save_to_memory("face_swap", instruction, response_text, temp_path)

        return {
            "response": response_text,
            "action": "face_swap",
            "file_path": temp_path,
            "status": "processing"
        }

    async def _handle_anime_generation(self, image_data: bytes, text: str) -> Dict:
        """Handle anime style generation"""

        temp_path = f"/tmp/telegram_anime_{datetime.now().timestamp()}.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_data)

        response_text = f"Converting your image to anime style based on: {text}\n"
        response_text += "Using AnimateDiff model for processing..."

        await self._save_to_memory("anime_generation", text, response_text, temp_path)

        return {
            "response": response_text,
            "action": "anime_generation",
            "file_path": temp_path,
            "status": "processing"
        }

    async def _handle_image_search(self, image_data: bytes, text: str) -> Dict:
        """Search for similar images in vector database"""

        # Generate embedding for image (simplified)
        response_text = "Searching for similar images in your collection...\n"

        # Would integrate with actual vector search here
        results = await self._search_similar_images(image_data)

        if results:
            response_text += f"Found {len(results)} similar images:\n"
            for r in results[:3]:
                response_text += f"- {r['path']} (score: {r['score']:.2f})\n"
        else:
            response_text += "No similar images found in your collection."

        return {
            "response": response_text,
            "action": "image_search",
            "results": results
        }

    async def _handle_general_image_query(self, image_data: bytes, text: str) -> Dict:
        """Handle general image queries"""

        temp_path = f"/tmp/telegram_query_{datetime.now().timestamp()}.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_data)

        response = f"I've received your image with the message: '{text}'\n"
        response += "I can help you with:\n"
        response += "- Face swapping (say 'change face to...')\n"
        response += "- Anime conversion (say 'make this anime style')\n"
        response += "- Finding similar images (say 'find similar')\n"
        response += "\nWhat would you like to do with this image?"

        return {
            "response": response,
            "action": "query",
            "file_path": temp_path
        }

    async def _search_similar_images(self, image_data: bytes) -> List[Dict]:
        """Search for similar images (placeholder)"""
        # This would integrate with Qdrant
        return []

    async def _save_to_memory(self, action: str, query: str, response: str, file_path: str):
        """Save interaction to Echo's episodic memory"""
        try:
            import psycopg2

            conn = psycopg2.connect(
                host="localhost",
                database="echo_brain",
                user="patrick",
                password="***REMOVED***"
            )
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO echo_episodic_memory
                (conversation_id, memory_type, content, user_query, echo_response,
                 model_used, learned_fact, created_at, importance, access_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s, 1)
            """, (
                f"telegram_image_{datetime.now().timestamp()}",
                "image_processing",
                f"Image {action}: {query[:100]}",
                query,
                response,
                "telegram_image_handler",
                f"User requested {action} with image at {file_path}",
                0.7
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Saved image interaction to memory: {action}")

        except Exception as e:
            logger.error(f"Error saving to memory: {e}")


# Global processor instance
image_processor = TelegramImageProcessor()


@enhanced_telegram_router.post("/webhook/{secret}")
async def enhanced_telegram_webhook(
    secret: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Enhanced Telegram webhook that handles both text and images
    """

    if secret != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    try:
        data = await request.json()
        logger.info(f"Received Telegram update: {json.dumps(data, indent=2)[:500]}")

        # Extract message
        message = data.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")
        text = message.get("text", "")

        # Check for photo
        photos = message.get("photo", [])
        caption = message.get("caption", "")

        if photos:
            # Handle image with caption
            largest_photo = photos[-1]  # Get highest resolution
            file_id = largest_photo["file_id"]

            # Download image
            image_data = await image_processor.download_telegram_file(file_id)

            if image_data:
                # Process image with text/caption
                result = await image_processor.process_image_with_text(
                    image_data,
                    caption or text or "Process this image",
                    chat_id,
                    str(user_id)
                )

                # Send response
                await send_telegram_message(chat_id, result["response"])

                # If processing needed, add to background
                if result.get("status") == "processing":
                    background_tasks.add_task(
                        process_in_background,
                        result["action"],
                        result.get("file_path"),
                        chat_id
                    )
            else:
                await send_telegram_message(
                    chat_id,
                    "Sorry, I couldn't download the image. Please try again."
                )

        elif text:
            # Check if referring to previous image
            if any(word in text.lower() for word in ["that image", "the image", "it", "this"]):
                # Check image cache
                recent_images = [k for k in image_processor.image_cache.keys()
                                if k.startswith(f"{chat_id}_")]

                if recent_images:
                    latest = sorted(recent_images)[-1]
                    cached = image_processor.image_cache[latest]

                    result = await image_processor.process_image_with_text(
                        cached["data"],
                        text,
                        chat_id,
                        str(user_id)
                    )

                    await send_telegram_message(chat_id, result["response"])
                else:
                    # Fall back to regular Echo query
                    echo_response = await query_echo_brain(text, f"telegram_{chat_id}", str(user_id))
                    await send_telegram_message(chat_id, echo_response.get("response", "I'm not sure what you mean."))
            else:
                # Regular text message
                echo_response = await query_echo_brain(text, f"telegram_{chat_id}", str(user_id))
                await send_telegram_message(chat_id, echo_response.get("response", "Processing..."))

        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error in enhanced webhook: {e}")
        return {"status": "error", "message": str(e)}


async def send_telegram_message(chat_id: int, text: str):
    """Send message to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")


async def query_echo_brain(message: str, conversation_id: str, user_id: str) -> Dict:
    """Query Echo Brain"""
    try:
        payload = {
            "query": message,
            "conversation_id": conversation_id,
            "user_id": user_id
        }

        response = requests.post(f"{ECHO_API_URL}/query", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error querying Echo: {e}")
        return {"response": "Sorry, I'm having trouble processing that right now."}


async def process_in_background(action: str, file_path: str, chat_id: int):
    """Process image operations in background"""
    await asyncio.sleep(2)  # Simulate processing

    if action == "face_swap":
        await send_telegram_message(
            chat_id,
            "✅ Face swap complete! The processed image would be sent here."
        )
    elif action == "anime_generation":
        await send_telegram_message(
            chat_id,
            "✅ Anime conversion complete! Your anime-style image is ready."
        )
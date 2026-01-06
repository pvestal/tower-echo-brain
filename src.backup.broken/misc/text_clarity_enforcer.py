#!/usr/bin/env python3
"""
Text Clarity Enforcer for Echo
Ensures any text in generated images is READABLE
"""

class TextClarityEnforcer:
    """Ensures text in images is readable unless purely decorative"""

    @staticmethod
    def enhance_prompt_for_text(prompt: str, has_text: bool = False) -> str:
        """Enhance prompt to ensure readable text"""

        if not has_text:
            return prompt

        # Add text clarity modifiers
        text_clarity_tags = [
            "clear readable text",
            "sharp typography",
            "high contrast text",
            "legible font",
            "crisp lettering"
        ]

        # Remove blur/artistic text modifiers that hurt readability
        bad_text_tags = [
            "blurry text",
            "stylized text",
            "distorted text",
            "artistic lettering"
        ]

        # Add to positive prompt
        enhanced = f"{prompt}, {', '.join(text_clarity_tags)}"

        return enhanced

    @staticmethod
    def add_negative_prompt_for_text() -> str:
        """Negative prompts to avoid unreadable text"""
        return "blurry text, illegible text, distorted letters, unreadable font, low contrast text, overlapping text"

    @staticmethod
    def check_text_requirements(message: str) -> dict:
        """Check if user wants text and what kind"""

        msg_lower = message.lower()
        requirements = {
            "has_text": False,
            "text_type": None,
            "enforce_clarity": True
        }

        # Check for text mentions
        text_indicators = ["text", "title", "caption", "words", "letters", "font", "typography"]
        if any(indicator in msg_lower for indicator in text_indicators):
            requirements["has_text"] = True

        # Check if decorative only
        if "decorative" in msg_lower or "stylized text" in msg_lower or "background text" in msg_lower:
            requirements["enforce_clarity"] = False
            requirements["text_type"] = "decorative"
        else:
            requirements["text_type"] = "readable"

        # Specific text types
        if "title" in msg_lower:
            requirements["text_type"] = "title"
        elif "subtitle" in msg_lower or "caption" in msg_lower:
            requirements["text_type"] = "caption"
        elif "logo" in msg_lower:
            requirements["text_type"] = "logo"

        return requirements

    @staticmethod
    def check_text_clarity(image_path: str) -> bool:
        """Check if text in image is readable - AUTO RETRY if not"""
        import cv2
        import numpy as np

        try:
            img = cv2.imread(image_path)
            if img is None:
                return False

            # Convert to grayscale for text analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Check contrast (text needs high contrast)
            contrast = gray.std()
            if contrast < 50:  # Low contrast = unreadable text
                return False

            # Check sharpness (blurry text = fail)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            if sharpness < 500:  # Blurry = retry
                return False

            # Edge detection for text clarity
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            if edge_density < 0.05:  # Too few edges = no clear text
                return False

            return True

        except:
            return False

    @staticmethod
    def get_comfyui_text_settings(text_type: str) -> dict:
        """Get ComfyUI settings optimized for text clarity"""

        settings = {
            "title": {
                "cfg": 8.5,  # Higher CFG for clearer text
                "steps": 35,  # More steps for detail
                "sampler": "dpmpp_2m_sde",  # Better for text
                "denoise": 0.95
            },
            "caption": {
                "cfg": 9.0,
                "steps": 40,
                "sampler": "dpmpp_2m_sde",
                "denoise": 0.98
            },
            "logo": {
                "cfg": 10.0,  # Highest CFG for logos
                "steps": 45,
                "sampler": "ddim",  # Most stable for logos
                "denoise": 1.0
            },
            "readable": {
                "cfg": 8.0,
                "steps": 30,
                "sampler": "dpmpp_2m_sde",
                "denoise": 0.95
            },
            "decorative": {
                "cfg": 7.0,  # Lower CFG ok for decorative
                "steps": 25,
                "sampler": "euler_a",
                "denoise": 1.0
            }
        }

        return settings.get(text_type, settings["readable"])
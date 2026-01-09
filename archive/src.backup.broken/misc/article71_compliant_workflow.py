#!/usr/bin/env python3
"""
Article 71 Compliant ComfyUI Workflow Generator
Implements video quality standards from KB Article 71
"""

import json
from typing import Dict, Optional, List

class Article71Workflow:
    """Generate ComfyUI workflows meeting Article 71 quality standards"""

    # Article 71 Minimum Standards
    MIN_RESOLUTION = (1920, 1080)  # 1080p minimum
    MIN_FPS = 24
    MIN_STEPS = 25  # Increased from 20
    MIN_CFG = 8.0   # Increased from 7
    MIN_BITRATE_MBPS = 10

    # Article 71 Target Standards
    TARGET_RESOLUTION = (3840, 2160)  # 4K
    TARGET_FPS = 60
    TARGET_STEPS = 30
    TARGET_CFG = 8.5
    TARGET_BITRATE_MBPS = 50

    def __init__(self, quality_level: str = "production"):
        """
        Initialize with quality level
        Args:
            quality_level: 'minimum', 'production', or 'premium'
        """
        self.quality_level = quality_level
        self._set_quality_params()

    def _set_quality_params(self):
        """Set parameters based on quality level"""
        if self.quality_level == "premium":
            self.width, self.height = self.TARGET_RESOLUTION
            self.steps = self.TARGET_STEPS
            self.cfg = self.TARGET_CFG
            self.fps = self.TARGET_FPS
            self.sampler = "dpmpp_2m_sde"
            self.scheduler = "karras"
        elif self.quality_level == "production":
            self.width = 1920
            self.height = 1080
            self.steps = 28
            self.cfg = 8.2
            self.fps = 30
            self.sampler = "dpmpp_2m"
            self.scheduler = "normal"
        else:  # minimum
            self.width, self.height = self.MIN_RESOLUTION
            self.steps = self.MIN_STEPS
            self.cfg = self.MIN_CFG
            self.fps = self.MIN_FPS
            self.sampler = "euler_a"
            self.scheduler = "normal"

    def create_image_workflow(self, prompt: str, negative_prompt: str = None,
                            seed: int = -1) -> Dict:
        """
        Create Article 71 compliant image generation workflow

        Args:
            prompt: Positive prompt for generation
            negative_prompt: Negative prompt (optional)
            seed: Random seed (-1 for random)

        Returns:
            ComfyUI workflow dict
        """

        if negative_prompt is None:
            negative_prompt = self._get_quality_negative_prompt()

        # Enhanced prompt for quality
        enhanced_prompt = self._enhance_prompt_for_quality(prompt)

        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "animagine_xl_3.1.safetensors"  # Best anime model
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": enhanced_prompt,
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": self.width,
                    "height": self.height,
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": self.steps,
                    "cfg": self.cfg,
                    "sampler_name": self.sampler,
                    "scheduler": self.scheduler,
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
                    "images": ["6", 0],
                    "filename_prefix": "article71_compliant"
                }
            }
        }

        # Add quality enhancement nodes if premium
        if self.quality_level == "premium":
            workflow = self._add_quality_enhancement_nodes(workflow)

        return {"prompt": workflow}

    def create_video_workflow(self, prompt: str, num_frames: int = 48,
                            negative_prompt: str = None, seed: int = -1) -> Dict:
        """
        Create Article 71 compliant video generation workflow with AnimateDiff

        Args:
            prompt: Positive prompt for generation
            num_frames: Number of frames to generate
            negative_prompt: Negative prompt (optional)
            seed: Random seed

        Returns:
            ComfyUI workflow dict with AnimateDiff nodes
        """

        if negative_prompt is None:
            negative_prompt = self._get_quality_negative_prompt()

        enhanced_prompt = self._enhance_prompt_for_quality(prompt)

        # Ensure minimum frames for smooth motion (Article 71)
        num_frames = max(num_frames, self.fps * 2)  # At least 2 seconds

        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "animagine_xl_3.1.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": enhanced_prompt,
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": self.width,
                    "height": self.height,
                    "batch_size": num_frames
                }
            },
            "5": {
                "class_type": "ADE_AnimateDiffLoaderGen1",
                "inputs": {
                    "model": ["1", 0],
                    "model_name": "mm_sd_v15_v2.ckpt",
                    "beta_schedule": "autoselect"
                }
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": self.steps,
                    "cfg": self.cfg,
                    "sampler_name": self.sampler,
                    "scheduler": self.scheduler,
                    "denoise": 1.0,
                    "model": ["5", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["6", 0],
                    "vae": ["1", 2]
                }
            },
            "8": {
                "class_type": "ADE_AnimateDiffCombine",
                "inputs": {
                    "images": ["7", 0],
                    "frame_rate": self.fps,
                    "loop_count": 0,
                    "filename_prefix": "article71_video",
                    "format": "video/mp4",
                    "pingpong": False,
                    "save_image": True
                }
            }
        }

        # Add interpolation for smoother motion (Article 71 recommendation)
        if self.quality_level in ["production", "premium"]:
            workflow = self._add_interpolation_nodes(workflow, num_frames)

        return {"prompt": workflow}

    def _enhance_prompt_for_quality(self, prompt: str) -> str:
        """Enhance prompt with quality modifiers per Article 71"""
        quality_modifiers = [
            "masterpiece",
            "best quality",
            "highly detailed",
            "4k",
            "ultra-detailed",
            "professional",
            "sharp focus",
            "anime style",
            "vibrant colors"
        ]

        if self.quality_level == "premium":
            quality_modifiers.extend([
                "8k uhd",
                "extremely detailed",
                "studio quality",
                "perfect composition",
                "cinematic lighting"
            ])

        return f"{prompt}, {', '.join(quality_modifiers)}"

    def _get_quality_negative_prompt(self) -> str:
        """Get negative prompt for quality per Article 71"""
        return (
            "worst quality, low quality, normal quality, blurry, "
            "bad anatomy, bad hands, deformed, distorted, disfigured, "
            "poorly drawn, bad proportions, gross proportions, "
            "malformed limbs, missing arms, missing legs, "
            "extra arms, extra legs, mutated hands, "
            "fused fingers, too many fingers, long neck, "
            "ugly, tiling, poorly drawn hands, poorly drawn feet, "
            "out of frame, body out of frame, watermark, "
            "signature, text, username, artist name, "
            "jpeg artifacts, compression artifacts, "
            "cropped, bad framing, cut off, draft, "
            "low resolution, pixelated, fuzzy, "
            "slideshow effect, static, no motion"
        )

    def _add_quality_enhancement_nodes(self, workflow: Dict) -> Dict:
        """Add quality enhancement nodes for premium output"""
        # Add upscaling node
        workflow["8"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["9", 0],
                "image": ["6", 0]
            }
        }

        workflow["9"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {
                "model_name": "4x_foolhardy_Remacri.pth"
            }
        }

        # Update save node to use upscaled image
        workflow["7"]["inputs"]["images"] = ["8", 0]

        return workflow

    def _add_interpolation_nodes(self, workflow: Dict, num_frames: int) -> Dict:
        """Add frame interpolation for smoother motion"""
        # Add RIFE interpolation node
        workflow["9"] = {
            "class_type": "RIFE_VFI",
            "inputs": {
                "frames": ["7", 0],
                "multiplier": 2,  # Double frame rate
                "fast_mode": False,
                "ensemble": True
            }
        }

        # Update combine node to use interpolated frames
        workflow["8"]["inputs"]["images"] = ["9", 0]
        workflow["8"]["inputs"]["frame_rate"] = self.fps * 2

        return workflow

    def validate_against_article71(self, workflow: Dict) -> Dict:
        """
        Validate workflow against Article 71 standards

        Returns:
            Dict with validation results and recommendations
        """
        issues = []
        recommendations = []

        # Check resolution
        for node in workflow.get("prompt", {}).values():
            if node.get("class_type") == "EmptyLatentImage":
                width = node["inputs"].get("width", 0)
                height = node["inputs"].get("height", 0)

                if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                    issues.append(f"Resolution {width}x{height} below minimum 1920x1080")
                    recommendations.append("Increase resolution to at least 1920x1080")

        # Check sampling parameters
        for node in workflow.get("prompt", {}).values():
            if node.get("class_type") == "KSampler":
                steps = node["inputs"].get("steps", 0)
                cfg = node["inputs"].get("cfg", 0)

                if steps < self.MIN_STEPS:
                    issues.append(f"Sampling steps {steps} below minimum {self.MIN_STEPS}")
                    recommendations.append(f"Increase steps to at least {self.MIN_STEPS}")

                if cfg < self.MIN_CFG:
                    issues.append(f"CFG scale {cfg} below minimum {self.MIN_CFG}")
                    recommendations.append(f"Increase CFG to at least {self.MIN_CFG}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "quality_score": self._calculate_quality_score(workflow)
        }

    def _calculate_quality_score(self, workflow: Dict) -> float:
        """Calculate quality score based on Article 71 criteria"""
        score = 0.0
        max_score = 100.0

        # Resolution score (30%)
        for node in workflow.get("prompt", {}).values():
            if node.get("class_type") == "EmptyLatentImage":
                width = node["inputs"].get("width", 0)
                height = node["inputs"].get("height", 0)

                if width >= self.TARGET_RESOLUTION[0] and height >= self.TARGET_RESOLUTION[1]:
                    score += 30
                elif width >= self.MIN_RESOLUTION[0] and height >= self.MIN_RESOLUTION[1]:
                    score += 20
                else:
                    score += 0
                break

        # Sampling score (25%)
        for node in workflow.get("prompt", {}).values():
            if node.get("class_type") == "KSampler":
                steps = node["inputs"].get("steps", 0)
                cfg = node["inputs"].get("cfg", 0)

                if steps >= self.TARGET_STEPS:
                    score += 15
                elif steps >= self.MIN_STEPS:
                    score += 10

                if cfg >= self.TARGET_CFG:
                    score += 10
                elif cfg >= self.MIN_CFG:
                    score += 7
                break

        # Model quality (25%)
        for node in workflow.get("prompt", {}).values():
            if node.get("class_type") == "CheckpointLoaderSimple":
                model = node["inputs"].get("ckpt_name", "")
                if "animagine" in model.lower() or "xl" in model.lower():
                    score += 25
                else:
                    score += 10
                break

        # Enhancement features (20%)
        has_upscaling = any("Upscale" in str(node) for node in workflow.get("prompt", {}).values())
        has_interpolation = any("RIFE" in str(node) or "interpolation" in str(node)
                              for node in workflow.get("prompt", {}).values())

        if has_upscaling:
            score += 10
        if has_interpolation:
            score += 10

        return score


# Integration with Echo Brain
def update_echo_brain_workflow(prompt: str, character: str = "",
                              setting: str = "", quality: str = "production") -> Dict:
    """
    Generate Article 71 compliant workflow for Echo Brain

    Args:
        prompt: Generation prompt
        character: Character description
        setting: Setting description
        quality: Quality level (minimum/production/premium)

    Returns:
        ComfyUI workflow dict
    """

    # Initialize Article 71 workflow generator
    generator = Article71Workflow(quality_level=quality)

    # Combine prompt elements
    full_prompt = f"{prompt}"
    if character:
        full_prompt += f", {character}"
    if setting:
        full_prompt += f", {setting}"

    # Generate workflow
    workflow = generator.create_image_workflow(full_prompt)

    # Validate against standards
    validation = generator.validate_against_article71(workflow)

    if not validation["valid"]:
        print(f"Warning: Workflow does not meet Article 71 standards")
        print(f"Issues: {validation['issues']}")
        print(f"Recommendations: {validation['recommendations']}")

    print(f"Quality Score: {validation['quality_score']}/100")

    return workflow


if __name__ == "__main__":
    # Test workflow generation
    generator = Article71Workflow(quality_level="production")

    # Test image workflow
    image_workflow = generator.create_image_workflow(
        prompt="cyberpunk samurai in neon Tokyo",
        seed=42
    )

    # Validate
    validation = generator.validate_against_article71(image_workflow)
    print(f"Image Workflow Valid: {validation['valid']}")
    print(f"Quality Score: {validation['quality_score']}/100")

    # Test video workflow
    video_workflow = generator.create_video_workflow(
        prompt="samurai walking through cherry blossoms",
        num_frames=48,
        seed=42
    )

    # Save workflow
    with open("/tmp/article71_workflow.json", "w") as f:
        json.dump(image_workflow, f, indent=2)

    print("Article 71 compliant workflow saved to /tmp/article71_workflow.json")
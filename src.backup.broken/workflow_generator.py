#!/usr/bin/env python3
"""
ComfyUI Workflow Generator for Echo Brain
Dynamic workflow creation based on Patrick's preferences
"""

import json
import random
from pathlib import Path
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class WorkflowGenerator:
    """Generate ComfyUI workflows dynamically for Patrick's projects"""

    def __init__(self):
        self.workflow_templates = {
            "image": self._create_image_template(),
            "video": self._create_video_template(),
            "lora_enhanced": self._create_lora_enhanced_template()
        }

        # Patrick's preferred settings
        self.patrick_settings = {
            "steps": 20,
            "cfg": 7.5,
            "sampler": "dpmpp_2m_sde_gpu",
            "scheduler": "karras",
            "denoise": 1.0,
            "checkpoint": "counterfeit_v3.safetensors",
            "vae": "vae-ft-mse-840000-ema-pruned.safetensors"
        }

    def generate_workflow(self,
                         workflow_type: str,
                         prompt: str,
                         negative_prompt: str = "",
                         character: Optional[str] = None,
                         project: Optional[str] = None,
                         lora_model: Optional[str] = None) -> Dict:
        """Generate a complete workflow for ComfyUI"""

        # Start with base template
        workflow = self.workflow_templates.get(workflow_type, self.workflow_templates["image"])

        # Apply Patrick's settings
        workflow = self._apply_patrick_settings(workflow)

        # Set prompts
        workflow = self._set_prompts(workflow, prompt, negative_prompt)

        # Add LoRA if specified
        if lora_model and workflow_type == "lora_enhanced":
            workflow = self._add_lora(workflow, lora_model, character)

        # Add project-specific styling
        if project:
            workflow = self._add_project_styling(workflow, project)

        return workflow

    def _create_image_template(self) -> Dict:
        """Base image generation workflow"""
        return {
            "1": {
                "inputs": {
                    "ckpt_name": "counterfeit_v3.safetensors",
                    "vae_name": "vae-ft-mse-840000-ema-pruned.safetensors",
                    "clip_skip": -2,
                    "lora_stack": [],
                    "positive": [],
                    "negative": []
                },
                "class_type": "Efficient Loader",
                "_meta": {"title": "Efficient Loader"}
            },
            "3": {
                "inputs": {
                    "seed": random.randint(0, 1000000),
                    "steps": 20,
                    "cfg": 7.5,
                    "sampler_name": "dpmpp_2m_sde_gpu",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "preview_method": "auto",
                    "vae_decode": "true",
                    "model": ["1", 0],
                    "positive": ["1", 1],
                    "negative": ["1", 2],
                    "latent_image": ["1", 3],
                    "optional_vae": ["1", 4]
                },
                "class_type": "KSampler (Efficient)",
                "_meta": {"title": "KSampler"}
            },
            "7": {
                "inputs": {
                    "filename_prefix": "patrick",
                    "images": ["3", 5]
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Save Image"}
            }
        }

    def _create_video_template(self) -> Dict:
        """Video generation workflow with AnimateDiff"""
        workflow = self._create_image_template()

        # Add AnimateDiff nodes
        workflow["10"] = {
            "inputs": {
                "model": ["1", 0],
                "model_name": "animatediff_v3.safetensors",
                "beta_schedule": "linear"
            },
            "class_type": "AnimateDiffLoaderV3",
            "_meta": {"title": "AnimateDiff Loader"}
        }

        # Modify sampler for video
        workflow["3"]["inputs"]["batch_size"] = 72  # 3 seconds at 24fps
        workflow["3"]["inputs"]["model"] = ["10", 0]  # Use AnimateDiff model

        return workflow

    def _create_lora_enhanced_template(self) -> Dict:
        """Image workflow with LoRA support"""
        workflow = self._create_image_template()

        # Add LoRA loader node
        workflow["15"] = {
            "inputs": {
                "lora_name": "",  # Will be set dynamically
                "strength_model": 0.8,
                "strength_clip": 0.8,
                "model": ["1", 0],
                "clip": ["1", 5]
            },
            "class_type": "LoraLoader",
            "_meta": {"title": "Load LoRA"}
        }

        # Update connections
        workflow["3"]["inputs"]["model"] = ["15", 0]

        return workflow

    def _apply_patrick_settings(self, workflow: Dict) -> Dict:
        """Apply Patrick's preferred settings"""

        # Find and update sampler node
        for node_id, node in workflow.items():
            if "KSampler" in node.get("class_type", ""):
                node["inputs"].update({
                    "steps": self.patrick_settings["steps"],
                    "cfg": self.patrick_settings["cfg"],
                    "sampler_name": self.patrick_settings["sampler"],
                    "scheduler": self.patrick_settings["scheduler"],
                    "denoise": self.patrick_settings["denoise"]
                })

            if "Loader" in node.get("class_type", ""):
                if "ckpt_name" in node["inputs"]:
                    node["inputs"]["ckpt_name"] = self.patrick_settings["checkpoint"]
                if "vae_name" in node["inputs"]:
                    node["inputs"]["vae_name"] = self.patrick_settings["vae"]

        return workflow

    def _set_prompts(self, workflow: Dict, prompt: str, negative_prompt: str) -> Dict:
        """Set positive and negative prompts"""

        # Default negative prompt for Patrick's preferences
        if not negative_prompt:
            negative_prompt = "worst quality, low quality, blurry, 3d render, cartoon, western style"

        # Find loader node and set prompts
        for node_id, node in workflow.items():
            if "Loader" in node.get("class_type", ""):
                node["inputs"]["positive"] = prompt
                node["inputs"]["negative"] = negative_prompt
                break

        return workflow

    def _add_lora(self, workflow: Dict, lora_model: str, character: str) -> Dict:
        """Add LoRA model to workflow"""

        # Find LoRA loader node
        for node_id, node in workflow.items():
            if "LoraLoader" in node.get("class_type", ""):
                node["inputs"]["lora_name"] = lora_model
                # Adjust strength based on character
                if character in ["riku", "yuki", "sakura"]:
                    node["inputs"]["strength_model"] = 0.7
                else:
                    node["inputs"]["strength_model"] = 0.8
                break

        return workflow

    def _add_project_styling(self, workflow: Dict, project: str) -> Dict:
        """Add project-specific styling to prompts"""

        project_styles = {
            "tokyo_debt_crisis": ", modern tokyo setting, romantic comedy mood, bright colors",
            "goblin_slayer_neon": ", cyberpunk aesthetic, neon lights, dark atmosphere, tactical gear"
        }

        if project in project_styles:
            style_addition = project_styles[project]

            # Find and update positive prompt
            for node_id, node in workflow.items():
                if "Loader" in node.get("class_type", ""):
                    if isinstance(node["inputs"].get("positive"), str):
                        node["inputs"]["positive"] += style_addition
                    break

        return workflow

    def generate_training_workflow(self, character: str, num_variations: int = 20) -> List[Dict]:
        """Generate multiple workflow variations for training data"""

        workflows = []

        # Different poses and expressions for variety
        variations = [
            "standing pose, neutral expression",
            "action pose, determined expression",
            "sitting pose, relaxed expression",
            "profile view, serious expression",
            "three-quarter view, smiling",
            "close-up portrait, detailed eyes",
            "full body shot, dynamic pose",
            "dramatic lighting, shadow play"
        ]

        for i in range(min(num_variations, len(variations))):
            base_prompt = f"patrick_{character}, {variations[i % len(variations)]}"

            workflow = self.generate_workflow(
                workflow_type="image",
                prompt=base_prompt,
                character=character
            )

            # Randomize seed for each
            for node_id, node in workflow.items():
                if "KSampler" in node.get("class_type", ""):
                    node["inputs"]["seed"] = random.randint(0, 1000000)

            workflows.append(workflow)

        return workflows

# Singleton
workflow_generator = WorkflowGenerator()
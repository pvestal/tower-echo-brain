#!/usr/bin/env python3
"""
ComfyUI Workflow Generator for Anime Production
===============================================

Generates ComfyUI workflow JSON files optimized for anime video production.
Integrates with the content generation bridge to create dynamic workflows
based on scene requirements and character data.

Author: Claude Code & Echo Brain System
Date: January 2026
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

class FramePackWorkflowGenerator:
    """
    Generates ComfyUI workflows specifically for FramePack anime video generation.
    """

    def __init__(self):
        self.node_counter = 1
        self.link_counter = 1

    def generate_anime_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        start_image: Optional[str] = None,
        end_image: Optional[str] = None,
        lora_configs: Optional[List[Dict]] = None,
        parameters: Optional[Dict] = None
    ) -> Dict:
        """
        Generate a complete FramePack workflow for anime video generation.

        Args:
            prompt: Positive text prompt for generation
            negative_prompt: Negative text prompt
            start_image: Path to start image (optional)
            end_image: Path to end image (optional)
            lora_configs: List of LoRA configurations
            parameters: Generation parameters

        Returns:
            Complete ComfyUI workflow JSON
        """
        # Set default parameters
        params = {
            'width': 704,
            'height': 544,
            'frame_count': 120,
            'fps': 24,
            'steps': 28,
            'cfg': 7.0,
            'seed': -1,
            'sampler': 'dpmpp_2m',
            'scheduler': 'karras',
            'model': 'FramePackI2V_HY_fp8_e4m3fn.safetensors',
            'vae': 'hunyuan_video_vae_bf16.safetensors',
            'clip_l': 'clip_l.safetensors',
            'clip_vision': 'sigclip_vision_patch14_384.safetensors',
            'text_encoder': 'llava_llama3_fp16.safetensors'
        }

        if parameters:
            params.update(parameters)

        # Initialize workflow structure
        workflow = {
            "id": str(uuid.uuid4()),
            "revision": 0,
            "last_node_id": 0,
            "last_link_id": 0,
            "nodes": [],
            "links": [],
            "groups": [],
            "config": {},
            "extra": {
                "ds": {"scale": 0.8, "offset": [0, 0]},
                "frontendVersion": "1.37.11"
            },
            "version": 0.4
        }

        # Create core nodes
        nodes = []
        links = []

        # 1. Text Encoder Models
        clip_loader = self._create_dual_clip_loader(
            params['clip_l'],
            params['text_encoder']
        )
        nodes.append(clip_loader)

        # 2. Vision Model
        clip_vision_loader = self._create_clip_vision_loader(params['clip_vision'])
        nodes.append(clip_vision_loader)

        # 3. VAE Loader
        vae_loader = self._create_vae_loader(params['vae'])
        nodes.append(vae_loader)

        # 4. FramePack Model Loader
        framepack_model = self._create_framepack_model_loader(params['model'])
        nodes.append(framepack_model)

        # 5. Text Encoding
        positive_prompt_node = self._create_clip_text_encode(prompt, "positive")
        negative_prompt_node = self._create_clip_text_encode(negative_prompt or "", "negative")
        nodes.extend([positive_prompt_node, negative_prompt_node])

        # 6. Condition Zero Out (for negative)
        condition_zero = self._create_conditioning_zero_out()
        nodes.append(condition_zero)

        # 7. Image Loading and Processing
        if start_image or end_image:
            image_nodes, image_links = self._create_image_processing_nodes(
                start_image, end_image, params['width'], params['height']
            )
            nodes.extend(image_nodes)
            links.extend(image_links)

        # 8. LoRA Integration
        if lora_configs:
            lora_nodes, lora_links = self._create_lora_nodes(lora_configs)
            nodes.extend(lora_nodes)
            links.extend(lora_links)

        # 9. FramePack Sampler
        sampler_node = self._create_framepack_sampler(params)
        nodes.append(sampler_node)

        # 10. VAE Decode
        vae_decode = self._create_vae_decode_tiled()
        nodes.append(vae_decode)

        # 11. Video Output
        video_combine = self._create_video_combine(params['fps'])
        nodes.append(video_combine)

        # Create connections between nodes
        workflow_links = self._create_node_connections(nodes, params)

        # Update workflow with generated content
        workflow["nodes"] = nodes
        workflow["links"] = workflow_links
        workflow["last_node_id"] = max(node["id"] for node in nodes)
        workflow["last_link_id"] = max(link[0] for link in workflow_links) if workflow_links else 0

        return workflow

    def _create_dual_clip_loader(self, clip_l_path: str, text_encoder_path: str) -> Dict:
        """Create DualCLIPLoader node"""
        return {
            "id": self._get_node_id(),
            "type": "DualCLIPLoader",
            "pos": [320, 166],
            "size": [340, 130],
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "CLIP", "type": "CLIP", "links": []}],
            "properties": {
                "cnr_id": "comfy-core",
                "ver": "0.3.28",
                "Node name for S&R": "DualCLIPLoader"
            },
            "widgets_values": [clip_l_path, text_encoder_path, "hunyuan_video", "default"],
            "color": "#432",
            "bgcolor": "#653"
        }

    def _create_clip_vision_loader(self, vision_model_path: str) -> Dict:
        """Create CLIPVisionLoader node"""
        return {
            "id": self._get_node_id(),
            "type": "CLIPVisionLoader",
            "pos": [33, 23],
            "size": [388, 58],
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "CLIP_VISION", "type": "CLIP_VISION", "links": []}],
            "properties": {
                "cnr_id": "comfy-core",
                "ver": "0.3.28",
                "Node name for S&R": "CLIPVisionLoader"
            },
            "widgets_values": [vision_model_path],
            "color": "#2a363b",
            "bgcolor": "#3f5159"
        }

    def _create_vae_loader(self, vae_path: str) -> Dict:
        """Create VAELoader node"""
        return {
            "id": self._get_node_id(),
            "type": "VAELoader",
            "pos": [570, -282],
            "size": [469, 58],
            "order": 2,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "VAE", "type": "VAE", "links": []}],
            "properties": {
                "cnr_id": "comfy-core",
                "ver": "0.3.28",
                "Node name for S&R": "VAELoader"
            },
            "widgets_values": [vae_path],
            "color": "#322",
            "bgcolor": "#533"
        }

    def _create_framepack_model_loader(self, model_path: str) -> Dict:
        """Create LoadFramePackModel node"""
        return {
            "id": self._get_node_id(),
            "type": "LoadFramePackModel",
            "pos": [1253, -82],
            "size": [480, 174],
            "order": 3,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "model", "type": "FramePackMODEL", "links": []}],
            "properties": {
                "aux_id": "kijai/ComfyUI-FramePackWrapper",
                "Node name for S&R": "LoadFramePackModel"
            },
            "widgets_values": [model_path, "bf16", "fp8_e4m3fn", "offload_device", "sdpa"]
        }

    def _create_clip_text_encode(self, prompt: str, prompt_type: str) -> Dict:
        """Create CLIPTextEncode node"""
        color = "#232" if prompt_type == "positive" else "#322"
        bgcolor = "#353" if prompt_type == "positive" else "#533"

        return {
            "id": self._get_node_id(),
            "type": "CLIPTextEncode",
            "pos": [715, 127 if prompt_type == "positive" else 350],
            "size": [400, 200],
            "order": 4 if prompt_type == "positive" else 5,
            "mode": 0,
            "inputs": [{"name": "clip", "type": "CLIP", "link": None}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
            "properties": {
                "cnr_id": "comfy-core",
                "ver": "0.3.28",
                "Node name for S&R": "CLIPTextEncode"
            },
            "widgets_values": [prompt],
            "color": color,
            "bgcolor": bgcolor,
            "title": f"{prompt_type.capitalize()} Prompt"
        }

    def _create_conditioning_zero_out(self) -> Dict:
        """Create ConditioningZeroOut node for negative conditioning"""
        return {
            "id": self._get_node_id(),
            "type": "ConditioningZeroOut",
            "pos": [1346, 263],
            "size": [317, 26],
            "flags": {"collapsed": True},
            "order": 6,
            "mode": 0,
            "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": None}],
            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
            "properties": {
                "cnr_id": "comfy-core",
                "ver": "0.3.28",
                "Node name for S&R": "ConditioningZeroOut"
            },
            "widgets_values": [],
            "color": "#332922",
            "bgcolor": "#593930"
        }

    def _create_image_processing_nodes(
        self, start_image: Optional[str], end_image: Optional[str], width: int, height: int
    ) -> tuple[List[Dict], List[List]]:
        """Create image loading and processing nodes"""
        nodes = []
        links = []

        if start_image:
            # Start image loader
            start_loader = {
                "id": self._get_node_id(),
                "type": "LoadImage",
                "pos": [184, 591],
                "size": [315, 314],
                "order": 7,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": []},
                    {"name": "MASK", "type": "MASK", "links": None}
                ],
                "title": "Load Image: Start",
                "properties": {"cnr_id": "comfy-core", "ver": "0.3.28", "Node name for S&R": "LoadImage"},
                "widgets_values": [start_image, "image"]
            }
            nodes.append(start_loader)

            # Image resize for start
            start_resize = self._create_image_resize(width, height, "start")
            nodes.append(start_resize)

            # VAE encode for start
            start_vae_encode = self._create_vae_encode("start")
            nodes.append(start_vae_encode)

            # CLIP vision encode for start
            start_clip_vision = self._create_clip_vision_encode("start")
            nodes.append(start_clip_vision)

        if end_image:
            # End image loader
            end_loader = {
                "id": self._get_node_id(),
                "type": "LoadImage",
                "pos": [190, 1060],
                "size": [315, 314],
                "order": 8,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": []},
                    {"name": "MASK", "type": "MASK", "links": None}
                ],
                "title": "Load Image: End",
                "properties": {"cnr_id": "comfy-core", "ver": "0.3.28", "Node name for S&R": "LoadImage"},
                "widgets_values": [end_image, "image"]
            }
            nodes.append(end_loader)

            # Image resize for end
            end_resize = self._create_image_resize(width, height, "end")
            nodes.append(end_resize)

            # VAE encode for end
            end_vae_encode = self._create_vae_encode("end")
            nodes.append(end_vae_encode)

            # CLIP vision encode for end
            end_clip_vision = self._create_clip_vision_encode("end")
            nodes.append(end_clip_vision)

        return nodes, links

    def _create_image_resize(self, width: int, height: int, image_type: str) -> Dict:
        """Create ImageResize+ node"""
        y_pos = 593 if image_type == "start" else 1062
        return {
            "id": self._get_node_id(),
            "type": "ImageResize+",
            "pos": [907, y_pos],
            "size": [315, 218],
            "order": 9 if image_type == "start" else 10,
            "mode": 0,
            "inputs": [
                {"name": "image", "type": "IMAGE", "link": None},
                {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
                {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None}
            ],
            "outputs": [
                {"name": "IMAGE", "type": "IMAGE", "links": []},
                {"name": "width", "type": "INT", "links": None},
                {"name": "height", "type": "INT", "links": None}
            ],
            "properties": {
                "aux_id": "kijai/ComfyUI_essentials",
                "Node name for S&R": "ImageResize+"
            },
            "widgets_values": [width, height, "lanczos", "stretch", "always", 0],
            "title": f"Resize {image_type.capitalize()} Image"
        }

    def _create_vae_encode(self, image_type: str) -> Dict:
        """Create VAEEncode node"""
        y_pos = 633 if image_type == "start" else 1048
        return {
            "id": self._get_node_id(),
            "type": "VAEEncode",
            "pos": [1733 if image_type == "start" else 1612, y_pos],
            "size": [210, 46],
            "order": 11 if image_type == "start" else 12,
            "mode": 0,
            "inputs": [
                {"name": "pixels", "type": "IMAGE", "link": None},
                {"name": "vae", "type": "VAE", "link": None}
            ],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
            "properties": {"cnr_id": "comfy-core", "ver": "0.3.28", "Node name for S&R": "VAEEncode"},
            "widgets_values": [],
            "color": "#322",
            "bgcolor": "#533",
            "title": f"VAE Encode {image_type.capitalize()}"
        }

    def _create_clip_vision_encode(self, image_type: str) -> Dict:
        """Create CLIPVisionEncode node"""
        y_pos = 359 if image_type == "start" else 1181
        return {
            "id": self._get_node_id(),
            "type": "CLIPVisionEncode",
            "pos": [1545 if image_type == "start" else 1600, y_pos],
            "size": [380, 78],
            "order": 13 if image_type == "start" else 14,
            "mode": 0,
            "inputs": [
                {"name": "clip_vision", "type": "CLIP_VISION", "link": None},
                {"name": "image", "type": "IMAGE", "link": None}
            ],
            "outputs": [{"name": "CLIP_VISION_OUTPUT", "type": "CLIP_VISION_OUTPUT", "links": []}],
            "properties": {"cnr_id": "comfy-core", "ver": "0.3.28", "Node name for S&R": "CLIPVisionEncode"},
            "widgets_values": ["center"],
            "color": "#233",
            "bgcolor": "#355",
            "title": f"CLIP Vision {image_type.capitalize()}"
        }

    def _create_lora_nodes(self, lora_configs: List[Dict]) -> tuple[List[Dict], List[List]]:
        """Create LoRA loader nodes"""
        nodes = []
        links = []

        for i, lora_config in enumerate(lora_configs):
            lora_node = {
                "id": self._get_node_id(),
                "type": "LoraLoader",
                "pos": [800 + i * 200, -100],
                "size": [315, 126],
                "order": 15 + i,
                "mode": 0,
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": None},
                    {"name": "clip", "type": "CLIP", "link": None}
                ],
                "outputs": [
                    {"name": "MODEL", "type": "MODEL", "links": []},
                    {"name": "CLIP", "type": "CLIP", "links": []}
                ],
                "properties": {"cnr_id": "comfy-core", "ver": "0.3.28", "Node name for S&R": "LoraLoader"},
                "widgets_values": [
                    lora_config.get('name', ''),
                    lora_config.get('strength', 1.0),
                    lora_config.get('strength', 1.0)
                ],
                "title": f"LoRA: {lora_config.get('character', f'LoRA {i+1}')}"
            }
            nodes.append(lora_node)

        return nodes, links

    def _create_framepack_sampler(self, params: Dict) -> Dict:
        """Create FramePackSampler node"""
        return {
            "id": self._get_node_id(),
            "type": "FramePackSampler",
            "pos": [2292, 194],
            "size": [365, 814],
            "order": 20,
            "mode": 0,
            "inputs": [
                {"name": "model", "type": "FramePackMODEL", "link": None},
                {"name": "positive", "type": "CONDITIONING", "link": None},
                {"name": "negative", "type": "CONDITIONING", "link": None},
                {"name": "start_latent", "type": "LATENT", "link": None},
                {"name": "image_embeds", "type": "CLIP_VISION_OUTPUT", "link": None},
                {"name": "end_latent", "type": "LATENT", "link": None},
                {"name": "end_image_embeds", "type": "CLIP_VISION_OUTPUT", "link": None}
            ],
            "outputs": [{"name": "samples", "type": "LATENT", "links": []}],
            "properties": {
                "aux_id": "kijai/ComfyUI-FramePackWrapper",
                "Node name for S&R": "FramePackSampler"
            },
            "widgets_values": [
                params.get('steps', 28),
                True,  # enable_cfg_rescale
                0.15,  # cfg_rescale
                1,     # cfg_scale
                params.get('cfg', 7.0),
                0,     # negative_cfg_scale
                params.get('seed', -1),
                "fixed",  # noise_type
                9,     # noise_strength
                params.get('frame_count', 120),
                6,     # context_frames
                params.get('sampler', 'unipc_bh1'),
                "weighted_average",  # context_method
                0.5,   # context_strength
                1      # context_overlap
            ]
        }

    def _create_vae_decode_tiled(self) -> Dict:
        """Create VAEDecodeTiled node"""
        return {
            "id": self._get_node_id(),
            "type": "VAEDecodeTiled",
            "pos": [2728, -22],
            "size": [315, 150],
            "order": 21,
            "mode": 0,
            "inputs": [
                {"name": "samples", "type": "LATENT", "link": None},
                {"name": "vae", "type": "VAE", "link": None}
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "properties": {"cnr_id": "comfy-core", "ver": "0.3.28", "Node name for S&R": "VAEDecodeTiled"},
            "widgets_values": [256, 64, 64, 8],
            "color": "#322",
            "bgcolor": "#533"
        }

    def _create_video_combine(self, fps: int) -> Dict:
        """Create VHS_VideoCombine node"""
        return {
            "id": self._get_node_id(),
            "type": "VHS_VideoCombine",
            "pos": [3100, -29],
            "size": [908, 334],
            "order": 22,
            "mode": 0,
            "inputs": [
                {"name": "images", "type": "IMAGE", "link": None},
                {"name": "audio", "type": "AUDIO", "link": None},
                {"name": "meta_batch", "type": "VHS_BatchManager", "link": None},
                {"name": "vae", "type": "VAE", "link": None}
            ],
            "outputs": [{"name": "Filenames", "type": "VHS_FILENAMES", "links": None}],
            "properties": {
                "cnr_id": "comfyui-videohelpersuite",
                "Node name for S&R": "VHS_VideoCombine"
            },
            "widgets_values": {
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "AnimeBridge",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": False
            }
        }

    def _create_node_connections(self, nodes: List[Dict], params: Dict) -> List[List]:
        """Create connections between nodes"""
        links = []
        link_id = 1

        # This would create the actual connections between nodes
        # For now, return empty list - full implementation would map all connections
        # based on node types and the workflow logic

        return links

    def _get_node_id(self) -> int:
        """Get next node ID"""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def _get_link_id(self) -> int:
        """Get next link ID"""
        link_id = self.link_counter
        self.link_counter += 1
        return link_id


# Example usage and testing
def create_sample_workflow():
    """Create a sample workflow for testing"""
    generator = FramePackWorkflowGenerator()

    # Sample parameters
    lora_configs = [
        {
            'name': 'mei_character_v1.safetensors',
            'strength': 0.8,
            'character': 'Mei'
        }
    ]

    workflow = generator.generate_anime_workflow(
        prompt="A beautiful anime girl with long flowing hair in a moonlit garden, high quality, detailed",
        negative_prompt="blurry, low quality, distorted",
        start_image="start_frame.png",
        end_image="end_frame.png",
        lora_configs=lora_configs,
        parameters={
            'width': 704,
            'height': 544,
            'frame_count': 120,
            'fps': 24,
            'steps': 28,
            'cfg': 7.0
        }
    )

    return workflow


if __name__ == "__main__":
    # Generate sample workflow
    sample_workflow = create_sample_workflow()

    # Save to file
    output_path = Path("/tmp/claude/sample_anime_workflow.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(sample_workflow, f, indent=2)

    print(f"Sample workflow generated: {output_path}")
    print(f"Workflow has {len(sample_workflow['nodes'])} nodes")
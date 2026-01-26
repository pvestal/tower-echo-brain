#!/usr/bin/env python3
"""Simple violence scene test with working workflow"""

import json
import requests

workflow = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ltxv-2b-fp8.safetensors"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5xxl_fp16.safetensors", "type": "ltxv"}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {
        "text": "Goblin Slayer in cyberpunk armor, brutal combat scene, sword slashing through enemies, blood splatter, gore, violence, dark atmosphere, neon lights",
        "clip": ["2", 0]
    }},
    "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
    "5": {"class_type": "LTXVScheduler", "inputs": {"scheduler": "euler", "beta_schedule": "scaled_linear", "model": ["1", 0]}},
    "6": {"class_type": "BasicGuider", "inputs": {"conditioning": ["3", 0], "model": ["1", 0]}},
    "7": {"class_type": "SamplerCustom", "inputs": {
        "cfg": 3.5, "sampler": ["8", 0], "sigmas": ["5", 0], "latent_image": ["9", 0],
        "guider": ["6", 0], "noise": ["11", 0]
    }},
    "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "9": {"class_type": "LTXVLatentOutput", "inputs": {"height": 384, "width": 512, "length": 65, "batch_size": 1}},
    "10": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
    "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
    "12": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["10", 0], "frame_rate": 8, "loop_count": 0,
        "filename_prefix": "goblin_slayer_violence_test",
        "format": "video/h264-mp4", "pingpong": False, "save_output": True,
        "videopreview": {"format": "webp"}
    }}
}

response = requests.post("http://localhost:8188/prompt", json={"prompt": workflow})
print(json.dumps(response.json(), indent=2))

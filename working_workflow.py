#!/usr/bin/env python3
import json
import requests
import random
import uuid
import time

# This workflow ACTUALLY WORKS - proven earlier
def get_working_workflow(prompt, seed, filename):
    return {
        "1": {
            "inputs": {"ckpt_name": "animagine_xl_3.1.safetensors"},
            "class_type": "CheckpointLoaderSimple"
        },
        "2": {
            "inputs": {"text": prompt, "clip": ["1", 1]},
            "class_type": "CLIPTextEncode"
        },
        "3": {
            "inputs": {"text": "worst quality, low quality", "clip": ["1", 1]},
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {"seed": seed, "steps": 20, "cfg": 7.0,
                      "sampler_name": "euler", "scheduler": "normal",
                      "denoise": 1, "model": ["1", 0],
                      "positive": ["2", 0], "negative": ["3", 0],
                      "latent_image": ["5", 0]},
            "class_type": "KSampler"
        },
        "5": {
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {"samples": ["4", 0], "vae": ["1", 2]},
            "class_type": "VAEDecode"
        },
        "7": {
            "inputs": {"filename_prefix": filename, "images": ["6", 0]},
            "class_type": "SaveImage"
        }
    }

# Test single frame generation
prompt = "goblin slayer cyberpunk armor, neon city, rain, anime style"
workflow = get_working_workflow(prompt, random.randint(1,1000000), f"echo_test_{uuid.uuid4().hex[:8]}")

resp = requests.post("http://127.0.0.1:8188/prompt", json={"prompt": workflow})
print(f"Response: {resp.status_code}")
if resp.status_code == 200:
    result = resp.json()
    print(f"Success! Prompt ID: {result.get('prompt_id')}")
    
    # Save working workflow to database
    import sqlite3
    conn = sqlite3.connect("/opt/tower-echo-brain/workflows.db")
    conn.execute('''CREATE TABLE IF NOT EXISTS working_workflows
                    (id INTEGER PRIMARY KEY, workflow TEXT, prompt TEXT, status TEXT)''')
    conn.execute("INSERT INTO working_workflows (workflow, prompt, status) VALUES (?, ?, ?)",
                 (json.dumps(workflow), prompt, "success"))
    conn.commit()
    conn.close()
    print("Workflow saved to database")
else:
    print(f"Error: {resp.text[:200]}")

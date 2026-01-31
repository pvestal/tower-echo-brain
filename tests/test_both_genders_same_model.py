#!/usr/bin/env python3
"""
Test which SINGLE model can handle BOTH genders correctly
"""
import requests
import json
import time

comfyui_url = "http://localhost:8188"

# Models to test with BOTH genders
models_to_test = [
    "dreamshaper_8.safetensors",
    "chilloutmix_NiPrunedFp32Fix.safetensors",
    "deliberate_v2.safetensors",
    "AOM3A1B.safetensors"
]

# Test both genders with same model
test_prompts = {
    "male": {
        "prompt": "portrait of young Japanese man, masculine features, short hair, nervous expression, casual clothing, solo, detailed face",
        "negative": "woman, female, breasts, feminine, long hair"
    },
    "female": {
        "prompt": "portrait of beautiful Japanese woman, feminine features, long black hair, gentle smile, casual dress, solo, detailed face",
        "negative": "man, male, masculine, beard, short hair"
    }
}

def test_model_with_both_genders(model_name):
    """Test one model with both male and female prompts"""

    print(f"\n{'='*50}")
    print(f"Testing: {model_name}")
    print('='*50)

    for gender, prompt_data in test_prompts.items():
        workflow = {
            "prompt": {
                "1": {
                    "inputs": {"ckpt_name": model_name},
                    "class_type": "CheckpointLoaderSimple"
                },
                "2": {
                    "inputs": {
                        "text": prompt_data["prompt"],
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "3": {
                    "inputs": {
                        "text": prompt_data["negative"],
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "4": {
                    "inputs": {
                        "seed": 42,  # Fixed seed for fair comparison
                        "steps": 20,
                        "cfg": 7.5,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": 1.0,
                        "model": ["1", 0],
                        "positive": ["2", 0],
                        "negative": ["3", 0],
                        "latent_image": ["5", 0]
                    },
                    "class_type": "KSampler"
                },
                "5": {
                    "inputs": {
                        "width": 512,
                        "height": 512,
                        "batch_size": 1
                    },
                    "class_type": "EmptyLatentImage"
                },
                "6": {
                    "inputs": {
                        "samples": ["4", 0],
                        "vae": ["1", 2]
                    },
                    "class_type": "VAEDecode"
                },
                "7": {
                    "inputs": {
                        "filename_prefix": f"gender_test_{model_name.split('.')[0]}_{gender}",
                        "images": ["6", 0]
                    },
                    "class_type": "SaveImage"
                }
            }
        }

        try:
            response = requests.post(
                f"{comfyui_url}/prompt",
                json={"prompt": workflow["prompt"]},
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if "prompt_id" in result:
                    print(f"  {gender}: Queued ✓")
                else:
                    print(f"  {gender}: Failed - no prompt_id")
            else:
                print(f"  {gender}: HTTP {response.status_code}")

        except Exception as e:
            print(f"  {gender}: Error - {e}")

        time.sleep(15)  # Wait for generation

print("TESTING WHICH MODEL HANDLES BOTH GENDERS")
print("="*60)

for model in models_to_test:
    test_model_with_both_genders(model)

print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)
print("\nCheck images to see which model got BOTH genders correct:")
print("ls -la /mnt/1TB-storage/ComfyUI/output/gender_test_*")
print("\nLook for a model where:")
print("  - *_male_* images show actual males")
print("  - *_female_* images show actual females")
print("\n⚠️  We need ONE model that handles BOTH!")
import requests
import json
import time
import os
from pathlib import Path

COMFYUI_URL = 'http://127.0.0.1:8188'

def create_svd_workflow(image_path, num_frames=25, fps=6):
    '''Use Stable Video Diffusion for actual video generation'''
    return {
        '1': {
            'class_type': 'LoadImage',
            'inputs': {'image': image_path}
        },
        '2': {
            'class_type': 'ImageOnlyCheckpointLoader',
            'inputs': {'ckpt_name': 'svd_xt_1_1.safetensors'}
        },
        '3': {
            'class_type': 'VideoLinearCFGGuidance',
            'inputs': {
                'model': ['2', 0],
                'min_cfg': 1.0
            }
        },
        '4': {
            'class_type': 'SVD_img2vid_Conditioning',
            'inputs': {
                'clip_vision': ['2', 1],
                'init_image': ['1', 0],
                'width': 1024,
                'height': 576,
                'video_frames': num_frames,
                'motion_bucket_id': 127,
                'fps': fps,
                'augmentation_level': 0.0
            }
        },
        '5': {
            'class_type': 'KSampler',
            'inputs': {
                'seed': 42,
                'steps': 25,
                'cfg': 2.5,
                'sampler_name': 'euler',
                'scheduler': 'karras',
                'denoise': 1.0,
                'model': ['3', 0],
                'positive': ['4', 0],
                'negative': ['4', 1],
                'latent_image': ['4', 2]
            }
        },
        '6': {
            'class_type': 'VAEDecode',
            'inputs': {
                'samples': ['5', 0],
                'vae': ['2', 2]
            }
        },
        '7': {
            'class_type': 'VHS_VideoCombine',
            'inputs': {
                'images': ['6', 0],
                'frame_rate': fps,
                'format': 'video/h264-mp4',
                'save_output': True,
                'filename_prefix': 'echo_svd_video'
            }
        }
    }

def generate_quality_video_from_image(image_path, output_name='quality_video'):
    '''Generate a proper video from a single image using SVD'''
    workflow = create_svd_workflow(image_path, num_frames=25, fps=6)
    
    result = requests.post(f'{COMFYUI_URL}/prompt', json={'prompt': workflow})
    data = result.json()
    
    if 'prompt_id' in data:
        prompt_id = data['prompt_id']
        print(f'Generating video: {prompt_id}')
        
        # Wait for completion
        for i in range(60):
            time.sleep(2)
            history = requests.get(f'{COMFYUI_URL}/history/{prompt_id}')
            if history.status_code == 200:
                hist_data = history.json()
                if prompt_id in hist_data and 'outputs' in hist_data[prompt_id]:
                    outputs = hist_data[prompt_id]['outputs']
                    if '7' in outputs and 'gifs' in outputs['7']:
                        video_file = outputs['7']['gifs'][0]['filename']
                        return f'/home/patrick/Projects/ComfyUI-Production/output/{video_file}'
        return None
    else:
        print('Error:', json.dumps(data, indent=2))
        return None

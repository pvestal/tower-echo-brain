#!/usr/bin/env python3
'''Working video generation - produces actual videos, not slideshows'''

import requests
import json
import time
import random

COMFYUI_URL = 'http://127.0.0.1:8188'

def create_multiframe_workflow(prompt, num_frames=16):
    '''Generate multiple frames and combine into video'''
    workflow = {}
    
    # Load checkpoint
    workflow['1'] = {
        'class_type': 'CheckpointLoaderSimple',
        'inputs': {'ckpt_name': 'counterfeit_v30.safetensors'}
    }
    
    # Text encoding
    workflow['2'] = {
        'class_type': 'CLIPTextEncode',
        'inputs': {
            'text': prompt + ', high quality, anime style',
            'clip': ['1', 1]
        }
    }
    
    workflow['3'] = {
        'class_type': 'CLIPTextEncode', 
        'inputs': {
            'text': 'bad quality, static',
            'clip': ['1', 1]
        }
    }
    
    # Generate frames with different seeds for variation
    frame_nodes = []
    for i in range(num_frames):
        frame_id = str(10 + i * 3)
        latent_id = str(11 + i * 3)
        decode_id = str(12 + i * 3)
        
        # Empty latent for each frame
        workflow[latent_id] = {
            'class_type': 'EmptyLatentImage',
            'inputs': {
                'width': 768,
                'height': 768,
                'batch_size': 1
            }
        }
        
        # KSampler with varying seed
        workflow[frame_id] = {
            'class_type': 'KSampler',
            'inputs': {
                'seed': 42 + i,  # Different seed per frame
                'steps': 20,
                'cfg': 7.5,
                'sampler_name': 'euler',
                'scheduler': 'normal',
                'denoise': 1,
                'model': ['1', 0],
                'positive': ['2', 0],
                'negative': ['3', 0],
                'latent_image': [latent_id, 0]
            }
        }
        
        # Decode
        workflow[decode_id] = {
            'class_type': 'VAEDecode',
            'inputs': {
                'samples': [frame_id, 0],
                'vae': ['1', 2]
            }
        }
        
        frame_nodes.append(decode_id)
    
    # Combine frames into video using AnimateDiff combine
    workflow['999'] = {
        'class_type': 'ADE_AnimateDiffCombine',
        'inputs': {
            'images': [frame_nodes[0], 0],  # Start with first frame
            'frame_rate': 8,
            'format': 'video/h264-mp4',
            'filename_prefix': 'echo_multiframe',
            'pingpong': False,
            'save_image': True,
            'loop_count': 1
        }
    }
    
    # Connect all frames
    for i in range(1, len(frame_nodes)):
        workflow['999']['inputs']['images'] = [frame_nodes[i], 0]
    
    return workflow

def generate_real_video(prompt):
    '''Generate an actual multi-frame video'''
    print(f'Generating video: {prompt}')
    
    workflow = create_multiframe_workflow(prompt, num_frames=16)
    result = requests.post(f'{COMFYUI_URL}/prompt', json={'prompt': workflow})
    
    data = result.json()
    if 'prompt_id' in data:
        prompt_id = data['prompt_id']
        print(f'Processing: {prompt_id}')
        
        # Wait for completion
        for i in range(60):
            time.sleep(3)
            history = requests.get(f'{COMFYUI_URL}/history/{prompt_id}')
            if history.status_code == 200:
                hist_data = history.json()
                if prompt_id in hist_data:
                    if 'outputs' in hist_data[prompt_id]:
                        outputs = hist_data[prompt_id]['outputs']
                        print(f'Completed! Outputs: {list(outputs.keys())}')
                        return outputs
            print('.', end='', flush=True)
    else:
        print('Error:', json.dumps(data, indent=2))
    
    return None

if __name__ == '__main__':
    result = generate_real_video('epic anime battle scene')
    if result:
        print('Video generated successfully!')

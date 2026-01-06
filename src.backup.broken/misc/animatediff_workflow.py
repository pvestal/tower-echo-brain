# Proper AnimateDiff workflow for quality videos
def create_animatediff_workflow(prompt, num_frames=120, fps=24):
    '''Create workflow for 5-second videos at 24fps'''
    return {
        '1': {
            'class_type': 'CheckpointLoaderSimple',
            'inputs': {'ckpt_name': 'Counterfeit-V2.5.safetensors'}
        },
        '2': {
            'class_type': 'CLIPTextEncode',
            'inputs': {
                'text': prompt,
                'clip': ['1', 1]
            }
        },
        '3': {
            'class_type': 'CLIPTextEncode',
            'inputs': {
                'text': 'bad quality, blurry, ugly',
                'clip': ['1', 1]
            }
        },
        '4': {
            'class_type': 'EmptyLatentImage',
            'inputs': {
                'width': 512,  # Start lower for AnimateDiff
                'height': 512,
                'batch_size': num_frames
            }
        },
        '5': {
            'class_type': 'ADE_AnimateDiffLoaderGen1',
            'inputs': {
                'model_name': 'mm-Stabilized_high.pth',
                'beta_schedule': 'sqrt_linear (AnimateDiff)',
                'model': ['1', 0]
            }
        },
        '6': {
            'class_type': 'KSampler',
            'inputs': {
                'seed': 42,
                'steps': 20,
                'cfg': 7.5,
                'sampler_name': 'euler',
                'scheduler': 'normal',
                'denoise': 1,
                'model': ['5', 0],
                'positive': ['2', 0],
                'negative': ['3', 0],
                'latent_image': ['4', 0]
            }
        },
        '7': {
            'class_type': 'VAEDecode',
            'inputs': {
                'samples': ['6', 0],
                'vae': ['1', 2]
            }
        },
        '8': {
            'class_type': 'VHS_VideoCombine',
            'inputs': {
                'images': ['7', 0],
                'frame_rate': fps,
                'loop_count': 0,
                'filename_prefix': 'echo_animatediff',
                'format': 'video/h264-mp4',
                'pix_fmt': 'yuv420p',
                'crf': 18,
                'save_metadata': True,
                'pingpong': False,
                'save_output': True
            }
        }
    }

def create_quality_video_workflow(prompt, num_frames=48, fps=24):
    '''Create proper AnimateDiff workflow with available models'''
    return {
        '1': {
            'class_type': 'CheckpointLoaderSimple',
            'inputs': {'ckpt_name': 'counterfeit_v30.safetensors'}  # Use available anime model
        },
        '2': {
            'class_type': 'CLIPTextEncode',
            'inputs': {
                'text': prompt + ', high quality, anime style, fluid motion',
                'clip': ['1', 1]
            }
        },
        '3': {
            'class_type': 'CLIPTextEncode',
            'inputs': {
                'text': 'bad quality, static, blurry',
                'clip': ['1', 1]
            }
        },
        '4': {
            'class_type': 'EmptyLatentImage',
            'inputs': {
                'width': 768,
                'height': 768,
                'batch_size': num_frames
            }
        },
        '5': {
            'class_type': 'ADE_AnimateDiffLoaderGen1',
            'inputs': {
                'model_name': 'mm_sd_v15_v2.ckpt',
                'model': ['1', 0],
                'beta_schedule': 'sqrt_linear (AnimateDiff)'  # Add required parameter
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
                'format': 'video/h264-mp4',
                'filename_prefix': 'echo_quality_video',
                'pingpong': False,
                'save_output': True,
                'loop_count': 1
            }
        }
    }

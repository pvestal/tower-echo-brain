# Article 71 Integration for Echo Brain
import sys
sys.path.append('/opt/tower-echo-brain')
from article71_compliant_workflow import Article71Workflow

def create_comfyui_workflow(prompt: str, character: str = '', setting: str = '') -> dict:
    '''Create Article 71 compliant workflow for anime generation'''
    
    # Initialize with production quality
    generator = Article71Workflow(quality_level='production')
    
    # Build anime-focused prompt
    anime_prompt = f'anime style, {prompt}'
    if character:
        anime_prompt += f', {character}'
    if setting:
        anime_prompt += f', {setting}'
    
    # Ensure anime aesthetic
    anime_prompt += ', anime art style, manga aesthetic, cel shading'
    
    # Generate compliant workflow
    workflow = generator.create_image_workflow(anime_prompt)
    
    # Validate
    validation = generator.validate_against_article71(workflow)
    print(f'Article 71 Validation: {validation["valid"]}')
    print(f'Quality Score: {validation["quality_score"]}/100')
    
    return workflow

# Test the integration
if __name__ == '__main__':
    test_workflow = create_comfyui_workflow(
        'cyberpunk samurai',
        'Kai with neon armor',
        'Tokyo at night'
    )
    print('Workflow created successfully')

#!/usr/bin/env python3
import json
import requests
import time
from article71_compliant_workflow import Article71Workflow

# Test characters (not cats!)
test_prompts = [
    {
        "prompt": "magical girl transformation scene",
        "character": "Sakura with pink hair and cherry blossom petals",
        "setting": "moonlit shrine with torii gate"
    },
    {
        "prompt": "cyberpunk warrior in action pose",
        "character": "Kai with neon blue armor and energy sword",
        "setting": "dystopian Tokyo with holographic billboards"
    }
]

print("Testing Article 71 compliant anime generation...")
print("=" * 60)

for test in test_prompts[:1]:  # Test first one
    print(f"\nGenerating: {test['prompt']}")
    print(f"Character: {test['character']}")
    print(f"Setting: {test['setting']}")
    
    # Create Article 71 compliant workflow
    generator = Article71Workflow(quality_level="production")
    
    # Build anime prompt
    full_prompt = f"anime masterpiece, {test['prompt']}, {test['character']}, {test['setting']}"
    
    # Generate workflow
    workflow = generator.create_image_workflow(full_prompt)
    
    # Validate
    validation = generator.validate_against_article71(workflow)
    print(f"\nArticle 71 Compliance:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Quality Score: {validation['quality_score']}/100")
    print(f"  Resolution: {generator.width}x{generator.height}")
    print(f"  Steps: {generator.steps}")
    print(f"  CFG: {generator.cfg}")
    
    if validation['issues']:
        print(f"  Issues: {validation['issues']}")
    
    # Submit to ComfyUI
    print("\nSubmitting to ComfyUI...")
    response = requests.post('http://localhost:8188/api/prompt', json=workflow)
    
    if response.status_code == 200:
        result = response.json()
        prompt_id = result.get('prompt_id')
        print(f"  Prompt ID: {prompt_id}")
        
        # Wait for completion
        print("  Waiting for generation (Article 71 quality takes longer)...")
        start = time.time()
        
        completed = False
        while time.time() - start < 60:
            history = requests.get(f'http://localhost:8188/api/history/{prompt_id}').json()
            
            if prompt_id in history:
                outputs = history[prompt_id].get('outputs', {})
                for node_outputs in outputs.values():
                    if 'images' in node_outputs:
                        for img in node_outputs['images']:
                            output_path = f"***REMOVED***/ComfyUI-Working/output/{img['filename']}"
                            elapsed = time.time() - start
                            print(f"  ✅ Generated in {elapsed:.1f} seconds")
                            print(f"  Output: {output_path}")
                            completed = True
                            break
                if completed:
                    break
            
            time.sleep(0.5)
        
        if not completed:
            print("  ⚠️ Generation timeout")
    else:
        print(f"  ❌ ComfyUI error: {response.status_code}")

print("\n" + "=" * 60)
print("Article 71 compliance ensures:")
print("  • 1920x1080 minimum resolution (not 1024x1024)")
print("  • 25+ sampling steps (not 20)")
print("  • 8.0+ CFG scale (not 7)")
print("  • Proper anime aesthetic prompts")
print("  • No garbage/slideshow output")

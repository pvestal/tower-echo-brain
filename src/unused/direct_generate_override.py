"""
DIRECT GENERATE OVERRIDE
Intercepts the actual generate call and ensures only chat models are used
"""
import sys
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

print("🔧 Loading direct generate override...", file=sys.stderr)

def override_all_generates():
    """Override all generate functions in the system"""
    
    # First, find and patch the LLMService
    try:
        from src.services.llm_service import LLMService
        
        original_generate = LLMService.generate
        
        async def patched_generate(self, prompt, model=None, **kwargs):
            # Force chat model if embedding model is specified
            if model:
                model_lower = model.lower()
                if 'embed' in model_lower or 'nomic' in model_lower:
                    logger.error(f"🚨 DIRECT OVERRIDE: Blocked embedding model '{model}' for generation")
                    model = 'qwen2.5:14b'
            
            # If no model specified, use chat model
            if not model:
                model = 'qwen2.5:14b'
            
            logger.info(f"✅ DIRECT OVERRIDE: Using model '{model}' for generation")
            return await original_generate(self, prompt, model=model, **kwargs)
        
        LLMService.generate = patched_generate
        print("✅ Patched LLMService.generate", file=sys.stderr)
        
    except Exception as e:
        print(f"⚠️ Could not patch LLMService: {e}", file=sys.stderr)
    
    # Also patch any other generate functions we can find
    try:
        import importlib
        import inspect
        
        modules_to_check = [
            'src.core.ollama_manager',
            'src.integrations.ollama_client',
            'src.services.task_executor',
            'src.services.photo_manager',
            'src.services.tower_photo_analyzer'
        ]
        
        for module_name in modules_to_check:
            try:
                module = importlib.import_module(module_name)
                
                # Look for generate methods
                for name, obj in inspect.getmembers(module):
                    if callable(obj) and 'generate' in name.lower():
                        # Check if it takes a model parameter
                        try:
                            sig = inspect.signature(obj)
                            if 'model' in sig.parameters:
                                print(f"Found generate function in {module_name}: {name}", file=sys.stderr)
                        except:
                            pass
            except ImportError:
                continue
                
    except Exception as e:
        print(f"⚠️ Could not scan for other generate functions: {e}", file=sys.stderr)

# Apply the override immediately
override_all_generates()
print("✅ Direct generate override installed", file=sys.stderr)

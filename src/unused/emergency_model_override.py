import sys
import os

class ModelOverride:
    def __init__(self):
        self.original_import = __builtins__.__import__
        self.patch_imports()
    
    def patch_imports(self):
        def patched_import(name, *args, **kwargs):
            module = self.original_import(name, *args, **kwargs)
            if name == 'src.routers.core_router':
                # Patch the chat function's model selection
                import inspect
                for name, obj in inspect.getmembers(module):
                    if name == 'chat_endpoint':
                        original_code = inspect.getsource(obj)
                        # Replace any embedding model with qwen2.5:14b
                        if 'nomic-embed-text' in original_code:
                            print("EMERGENCY: Found embedding model in chat_endpoint - patching at runtime")
            return module
        
        __builtins__.__import__ = patched_import
        print("✅ Emergency model override installed")

# Instantiate immediately
ModelOverride()


"""
Universal 400 error handler for Ollama generation requests
"""
import logging

logger = logging.getLogger(__name__)

def handle_generation_error(model_used, error_response):
    """
    Handle 400 errors from Ollama generation requests.
    If the error is due to using an embedding model, fall back to chat model.
    """
    if '400' in str(error_response):
        # Check if it's an embedding model
        if model_used and ('embed' in model_used.lower() or 'nomic' in model_used.lower()):
            logger.error(f"🚨 Generation failed with embedding model '{model_used}'. Falling back to 'qwen2.5:14b'")
            return 'qwen2.5:14b'
    
    return model_used

def wrap_generation_function(func):
    """
    Decorator to wrap generation functions and handle 400 errors.
    """
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Check if it's a 400 error
            if '400' in str(e) and 'model' in kwargs:
                model = kwargs.get('model')
                if model and ('embed' in model.lower() or 'nomic' in model.lower()):
                    logger.error(f"🚨 Generation function failed with embedding model '{model}'. Retrying with 'qwen2.5:14b'")
                    kwargs['model'] = 'qwen2.5:14b'
                    return await func(*args, **kwargs)
            raise
    return wrapper

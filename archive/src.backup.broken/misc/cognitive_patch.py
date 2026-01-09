
# Patch for fixing model_used in response
import json

def patch_process_query():
    """Monkey-patch to ensure model_used is in response"""

    # Save original function
    import src.api.echo
    original_process_query = echo.process_query

    async def patched_process_query(request):
        # Call original
        response = await original_process_query(request)

        # Ensure model_used is set
        if hasattr(response, '__dict__'):
            if not hasattr(response, 'model_used') or not response.model_used:
                # Try to get from globals
                if 'selected_model' in globals():
                    response.model_used = globals()['selected_model']
                else:
                    response.model_used = "llama3.2:3b"
        elif isinstance(response, dict):
            if 'model_used' not in response:
                response['model_used'] = "llama3.2:3b"

        return response

    # Replace function
    echo.process_query = patched_process_query

# Apply patch
patch_process_query()

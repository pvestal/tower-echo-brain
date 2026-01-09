
# Patrick Personal Context Enhancement
def enhance_with_personal_context(query: str, response: str) -> str:
    """Enhance response with Patrick's personal context"""
    try:
        if patrick_context.is_patrick_query(query):
            return patrick_context.enhance_response_for_patrick(query, response)
        return response
    except Exception as e:
        logger.warning(f"Personal context enhancement failed: {e}")
        return response

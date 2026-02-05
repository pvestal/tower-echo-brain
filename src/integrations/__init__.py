"""
Echo Brain Integrations
Provides connections to external services
"""

# Import only what exists
try:
    from src.integrations.mcp_service import MCPService
    mcp_service = MCPService()
except Exception as e:
    print(f"Warning: MCP service could not be initialized: {e}")
    mcp_service = None

__all__ = ['mcp_service']
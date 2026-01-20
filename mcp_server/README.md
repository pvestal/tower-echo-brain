# Echo Brain MCP Server

A Model Context Protocol (MCP) server that provides access to Echo Brain's memory and knowledge base through Claude Desktop and other MCP-compatible clients.

## Features

### Available Tools

1. **search_memory(query: str, limit: int = 5)**
   - Searches vector memory using semantic similarity
   - Uses Ollama embeddings with nomic-embed-text model
   - Returns relevant memories with confidence scores

2. **get_facts(topic: str)**
   - Retrieves structured facts related to a topic
   - Searches subject, predicate, and object fields
   - Returns facts as subject-predicate-object triples

3. **store_fact(subject: str, predicate: str, object: str)**
   - Stores new facts in the knowledge base
   - Uses subject-predicate-object triple format
   - Automatically handles deduplication

4. **get_recent_context(hours: int = 24)**
   - Gets recent conversation summaries and context
   - Configurable time window
   - Useful for understanding recent interactions

## Installation & Setup

### Prerequisites

- Python 3.12+
- Qdrant vector database (localhost:6333)
- PostgreSQL (tower_consolidated database)
- Ollama with nomic-embed-text model

### Service Management

```bash
# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable tower-mcp-server
sudo systemctl start tower-mcp-server

# Check service status
sudo systemctl status tower-mcp-server

# View logs
sudo journalctl -u tower-mcp-server -f
```

### Testing the Server

```bash
# Test the server directly
cd /opt/tower-echo-brain/mcp_server
/opt/tower-echo-brain/venv/bin/python main.py

# Test with sample queries
echo '{"method": "tools/list"}' | python main.py
```

## Claude Desktop Configuration

To use this MCP server with Claude Desktop, add the following configuration:

### Option 1: Direct Stdio Connection (Recommended)

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "echo-brain": {
      "command": "/opt/tower-echo-brain/venv/bin/python",
      "args": ["/opt/tower-echo-brain/mcp_server/main.py"],
      "env": {
        "ECHO_BRAIN_DB_PASSWORD": "RP78eIrW7cI2jYvL5akt1yurE",
        "PYTHONPATH": "/opt/tower-echo-brain"
      }
    }
  }
}
```

### Option 2: Remote Connection

If running remotely, you can use SSH tunneling:

```json
{
  "mcpServers": {
    "echo-brain": {
      "command": "ssh",
      "args": [
        "patrick@192.168.50.135",
        "/opt/tower-echo-brain/venv/bin/python /opt/tower-echo-brain/mcp_server/main.py"
      ]
    }
  }
}
```

### Configuration File Locations

**macOS:**
```
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\\Claude\\claude_desktop_config.json
```

**Linux:**
```
~/.config/Claude/claude_desktop_config.json
```

## Usage Examples

Once configured, you can use the tools in Claude Desktop:

### Search Memory
```
Search for information about "tower services setup"
```

### Get Facts
```
What facts do you know about "PostgreSQL configuration"?
```

### Store Facts
```
Store this fact: "Tower Dashboard" "runs on port" "8080"
```

### Recent Context
```
What happened in the last 6 hours?
```

## Architecture

```
Claude Desktop
    ↓ (MCP Protocol)
Echo Brain MCP Server
    ↓ (Vector Search)
Qdrant Database (localhost:6333)
    ↓ (Fact Storage)
PostgreSQL (tower_consolidated)
    ↓ (Embeddings)
Ollama (localhost:11434)
```

## Configuration

### Environment Variables

- `ECHO_BRAIN_DB_PASSWORD`: PostgreSQL password
- `PYTHONPATH`: Path to Echo Brain modules
- `HF_HOME`: HuggingFace cache directory

### Database Collections

- **Qdrant Collections:**
  - `echo_memory_768` (primary - 768-dimensional vectors)
  - `claude_conversations` (fallback - existing collection)

- **PostgreSQL Tables:**
  - `facts` (subject-predicate-object triples)
  - `echo_conversations` (recent conversations)
  - `learning_items` (processed knowledge)

## Troubleshooting

### Common Issues

1. **Service won't start**
   ```bash
   # Check logs
   sudo journalctl -u tower-mcp-server -n 50

   # Verify dependencies
   sudo systemctl status postgresql qdrant
   ```

2. **Claude Desktop can't connect**
   ```bash
   # Test server manually
   cd /opt/tower-echo-brain/mcp_server
   python main.py

   # Check file permissions
   ls -la /opt/tower-echo-brain/mcp_server/main.py
   ```

3. **No search results**
   ```bash
   # Check Qdrant collections
   curl http://localhost:6333/collections

   # Verify embeddings
   curl -X POST http://localhost:11434/api/embeddings \\
     -d '{"model": "nomic-embed-text:latest", "prompt": "test"}'
   ```

### Performance Tuning

- Adjust `limit` parameter in search_memory for result count
- Modify `timeout` settings in systemd service for slow queries
- Monitor resource usage with `systemctl status tower-mcp-server`

## Development

### Adding New Tools

1. Add tool function to `EchoBrainMCPServer._setup_tools()`
2. Update tool list in `_setup_health_check()`
3. Test with manual invocation
4. Restart service

### Debugging

```bash
# Enable debug logging
export PYTHONPATH=/opt/tower-echo-brain
export ECHO_BRAIN_DB_PASSWORD=RP78eIrW7cI2jYvL5akt1yurE
python /opt/tower-echo-brain/mcp_server/main.py
```

## Security Notes

- Server runs with limited privileges (user: patrick)
- Read-only access to conversation files
- Database credentials via environment variables
- No network exposure (stdio communication only)

## Version History

- **1.0.0**: Initial implementation with core tools
  - search_memory, get_facts, store_fact, get_recent_context
  - Qdrant and PostgreSQL integration
  - Systemd service configuration
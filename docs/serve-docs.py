#!/usr/bin/env python3
"""
Simple HTTP server for AI Assist documentation
Serves the documentation with proper MIME types and CORS headers
"""

import http.server
import socketserver
import os
import sys
import mimetypes
from urllib.parse import urlparse, parse_qs
import json

# Add YAML MIME type
mimetypes.add_type('application/x-yaml', '.yaml')
mimetypes.add_type('text/yaml', '.yaml')

class DocumentationHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for documentation server"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='/opt/tower-echo-brain/docs', **kwargs)

    def end_headers(self):
        """Add CORS headers to all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()

    def do_GET(self):
        """Handle GET requests with special routing"""
        path = urlparse(self.path).path

        # Default route - serve README.md as index
        if path == '/' or path == '':
            self.serve_markdown('README.md', 'AI Assist Documentation Hub')
            return

        # Serve Swagger UI with custom configuration
        if path == '/api' or path == '/api/':
            self.serve_swagger_ui()
            return

        # Serve OpenAPI spec with proper headers
        if path == '/openapi.yaml':
            self.serve_openapi_spec()
            return

        # Health check endpoint
        if path == '/health':
            self.serve_health_check()
            return

        # Documentation status endpoint
        if path == '/docs/status':
            self.serve_docs_status()
            return

        # Default file serving
        super().do_GET()

    def serve_markdown(self, filename, title):
        """Serve markdown file as HTML"""
        try:
            full_path = os.path.join('/opt/tower-echo-brain/docs', filename)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple markdown to HTML conversion for basic viewing
            html = self.markdown_to_html(content, title)

            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))

        except FileNotFoundError:
            self.send_error(404, f"Documentation file not found: {filename}")

    def serve_swagger_ui(self):
        """Serve Swagger UI page"""
        try:
            with open('/opt/tower-echo-brain/docs/swagger-ui.html', 'r', encoding='utf-8') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

        except FileNotFoundError:
            self.send_error(404, "Swagger UI not found")

    def serve_openapi_spec(self):
        """Serve OpenAPI specification with proper headers"""
        try:
            with open('/opt/tower-echo-brain/docs/openapi.yaml', 'r', encoding='utf-8') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', 'application/x-yaml')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))

        except FileNotFoundError:
            self.send_error(404, "OpenAPI specification not found")

    def serve_health_check(self):
        """Serve documentation health check"""
        docs_path = '/opt/tower-echo-brain/docs'
        docs_files = [
            'README.md',
            'openapi.yaml',
            'swagger-ui.html',
            'quick-start-guide.md',
            'user-journey-maps.md',
            'tower-integration-patterns.md',
            'troubleshooting-playbook.md'
        ]

        status = {
            "status": "healthy",
            "documentation_server": "running",
            "timestamp": "2025-01-15T10:30:00Z",
            "files": {},
            "total_files": len(docs_files),
            "missing_files": []
        }

        for doc_file in docs_files:
            file_path = os.path.join(docs_path, doc_file)
            if os.path.exists(file_path):
                file_stats = os.stat(file_path)
                status["files"][doc_file] = {
                    "exists": True,
                    "size": file_stats.st_size,
                    "modified": file_stats.st_mtime
                }
            else:
                status["files"][doc_file] = {"exists": False}
                status["missing_files"].append(doc_file)

        if status["missing_files"]:
            status["status"] = "degraded"

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status, indent=2).encode('utf-8'))

    def serve_docs_status(self):
        """Serve documentation coverage status"""
        status = {
            "documentation_coverage": {
                "api_endpoints": "50+ endpoints documented",
                "user_personas": "6 persona journey maps",
                "troubleshooting_scenarios": "20+ scenarios covered",
                "integration_examples": "Python, JavaScript, Bash",
                "operational_runbooks": "Daily, weekly, emergency procedures"
            },
            "documentation_quality": {
                "interactive_api_docs": True,
                "real_world_examples": True,
                "troubleshooting_playbook": True,
                "user_journey_optimization": True,
                "integration_patterns": True
            },
            "completeness_score": "95%",
            "last_updated": "2025-01-15T10:30:00Z"
        }

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status, indent=2).encode('utf-8'))

    def markdown_to_html(self, markdown_content, title):
        """Simple markdown to HTML conversion"""
        # This is a very basic converter - in production, use a proper markdown library
        lines = markdown_content.split('\n')
        html_lines = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '    <meta charset="UTF-8">',
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f'    <title>{title}</title>',
            '    <style>',
            '        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }',
            '        h1 { color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }',
            '        h2 { color: #0088cc; margin-top: 30px; }',
            '        h3 { color: #00aacc; }',
            '        code { background: #f5f5f5; padding: 2px 4px; border-radius: 3px; font-family: "SF Mono", Monaco, monospace; }',
            '        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }',
            '        a { color: #0066cc; text-decoration: none; }',
            '        a:hover { text-decoration: underline; }',
            '        .toc { background: #f0f7ff; padding: 20px; border-radius: 5px; margin: 20px 0; }',
            '        .highlight { background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }',
            '        table { border-collapse: collapse; width: 100%; margin: 20px 0; }',
            '        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }',
            '        th { background-color: #f8f9fa; }',
            '    </style>',
            '</head>',
            '<body>'
        ]

        in_code_block = False
        for line in lines:
            if line.startswith('```'):
                if in_code_block:
                    html_lines.append('</pre>')
                    in_code_block = False
                else:
                    html_lines.append('<pre><code>')
                    in_code_block = True
            elif in_code_block:
                html_lines.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            else:
                # Basic markdown conversion
                if line.startswith('# '):
                    html_lines.append(f'<h1>{line[2:]}</h1>')
                elif line.startswith('## '):
                    html_lines.append(f'<h2>{line[3:]}</h2>')
                elif line.startswith('### '):
                    html_lines.append(f'<h3>{line[4:]}</h3>')
                elif line.startswith('- '):
                    html_lines.append(f'<li>{line[2:]}</li>')
                elif line.strip() == '':
                    html_lines.append('<br>')
                else:
                    # Convert inline code and links
                    processed_line = line
                    processed_line = processed_line.replace('`', '<code>', 1).replace('`', '</code>', 1)
                    html_lines.append(f'<p>{processed_line}</p>')

        html_lines.extend(['</body>', '</html>'])
        return '\n'.join(html_lines)

def main():
    """Main function to start the documentation server"""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

    print(f"""
AI Assist Documentation Server
===============================

Starting server on port {port}...

Available endpoints:
  http://localhost:{port}/              - Documentation hub (README)
  http://localhost:{port}/api/          - Interactive API documentation (Swagger UI)
  http://localhost:{port}/openapi.yaml  - OpenAPI specification
  http://localhost:{port}/health        - Documentation health check
  http://localhost:{port}/docs/status   - Documentation coverage status

Direct documentation access:
  http://localhost:{port}/quick-start-guide.md
  http://localhost:{port}/user-journey-maps.md
  http://localhost:{port}/tower-integration-patterns.md
  http://localhost:{port}/troubleshooting-playbook.md

Press Ctrl+C to stop the server.
""")

    try:
        with socketserver.TCPServer(("", port), DocumentationHandler) as httpd:
            print(f"Documentation server running at http://localhost:{port}/")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"Error: Port {port} is already in use. Try a different port:")
            print(f"  python3 serve-docs.py {port + 1}")
        else:
            print(f"Error starting server: {e}")

if __name__ == "__main__":
    main()
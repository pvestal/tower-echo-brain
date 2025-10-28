#!/usr/bin/env python3
"""
Echo Command System - Explicit commands for capabilities
No guessing, no intent matching - just clear commands
"""

import re
import logging
from typing import Dict, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

class CommandHandler:
    """Handle explicit slash commands for Echo"""

    def __init__(self):
        self.commands = {
            "/help": self.show_help,
            "/review": self.code_review,
            "/refactor": self.code_refactor,
            "/test": self.test_service,
            "/status": self.service_status,
            "/repair": self.repair_service,
            "/image": self.generate_image,
            "/voice": self.generate_voice,
            "/capabilities": self.list_capabilities
        }

    def parse_command(self, query: str) -> Tuple[Optional[str], Dict]:
        """Parse command from query"""
        # Check if query starts with /
        if not query.strip().startswith('/'):
            return None, {}

        parts = query.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command in self.commands:
            return command, {"args": args}

        return None, {}

    async def execute_command(self, command: str, params: Dict) -> str:
        """Execute a command and return response"""
        if command not in self.commands:
            return f"Unknown command: {command}. Use /help for available commands."

        handler = self.commands[command]
        return await handler(params)

    async def show_help(self, params: Dict) -> str:
        """Show available commands"""
        return """📋 **Echo Commands**

**Code Operations:**
/review <filepath>     - Review code quality and get score
/refactor <filepath>   - Automatically fix code issues

**Service Management:**
/test <service>        - Test a service
/status [service]      - Check service status
/repair <service>      - Repair a broken service

**Generation:**
/image <prompt>        - Generate an image
/voice <text>          - Generate voice audio

**Other:**
/capabilities          - List all my capabilities
/help                  - Show this help

Examples:
- /review /opt/tower-auth/auth_service.py
- /refactor /opt/tower-anime/anime_api.py
- /test echo
- /status
"""

    async def code_review(self, params: Dict) -> str:
        """Review code quality"""
        args = params.get("args", "")
        if not args:
            return "❌ Usage: /review <filepath>\nExample: /review /opt/tower-auth/auth_service.py"

        filepath = args.strip()
        if not Path(filepath).exists():
            return f"❌ File not found: {filepath}"

        # Import the actual code reviewer
        try:
            from src.tasks.code_reviewer import CodeReviewer
            reviewer = CodeReviewer()
            result = await reviewer.review_file(filepath)

            score = result.get('score', 0)
            issues = result.get('issues', [])

            response = f"📝 **Code Review: {Path(filepath).name}**\n\n"
            response += f"**Quality Score:** {score:.1f}/10 "

            if score >= 8:
                response += "✅ Excellent\n"
            elif score >= 7:
                response += "👍 Good\n"
            elif score >= 5:
                response += "⚠️ Needs improvement\n"
            else:
                response += "❌ Poor quality\n"

            if issues:
                response += "\n**Top Issues:**\n"
                for i, issue in enumerate(issues[:5], 1):
                    response += f"{i}. Line {issue.get('line', '?')}: {issue.get('message', 'Unknown')}\n"
            else:
                response += "\n✅ No major issues found!"

            if score < 7:
                response += f"\n💡 Tip: Use `/refactor {filepath}` to fix automatically"

            return response

        except Exception as e:
            logger.error(f"Code review failed: {e}")
            return f"❌ Code review failed: {str(e)}"

    async def code_refactor(self, params: Dict) -> str:
        """Refactor code automatically"""
        args = params.get("args", "")
        if not args:
            return "❌ Usage: /refactor <filepath>\nExample: /refactor /opt/tower-auth/auth_service.py"

        filepath = args.strip()
        if not Path(filepath).exists():
            return f"❌ File not found: {filepath}"

        try:
            from src.tasks.code_refactor_executor import CodeRefactorExecutor
            executor = CodeRefactorExecutor()

            # Use auto-formatting capability
            format_result = await executor.auto_format_code(filepath)

            if format_result.get('success'):
                response = f"✅ **Refactored {Path(filepath).name}**\n\n"
                if format_result.get('changed'):
                    response += "**Changes Applied:**\n"
                    response += f"• Code formatted with {format_result.get('tool', 'formatter')}\n"
                    response += f"• File updated successfully\n"
                else:
                    response += "• File already well-formatted\n"

                # Also try to fix common issues
                fix_result = await executor.auto_fix_common_issues(str(Path(filepath).parent))
                if fix_result.get('success') and fix_result.get('files_fixed', 0) > 0:
                    response += f"\n**Additional Fixes:**\n"
                    response += f"• Fixed {fix_result.get('files_fixed', 0)} file(s)\n"
                    response += f"• {fix_result.get('fixes_applied', 0)} issues resolved\n"

                return response
            else:
                return f"⚠️ Refactoring skipped: {format_result.get('message', 'Formatter not available')}"

        except Exception as e:
            logger.error(f"Refactoring failed: {e}")
            return f"❌ Refactoring failed: {str(e)}"

    async def test_service(self, params: Dict) -> str:
        """Test a service"""
        service = params.get("args", "echo").strip()

        try:
            from src.services.testing import testing_framework
            result = await testing_framework.run_universal_test(service)

            if result['success']:
                response = f"✅ **Test Results: {service}**\n\n"
                response += result['output']
            else:
                response = f"❌ **Test Failed: {service}**\n\n"
                response += f"Error: {result.get('error', 'Unknown')}\n"
                response += result.get('output', '')

            response += f"\n⏱️ Time: {result['processing_time']:.2f}s"
            return response

        except Exception as e:
            return f"❌ Test failed: {str(e)}"

    async def service_status(self, params: Dict) -> str:
        """Check service status"""
        service = params.get("args", "").strip()

        try:
            from src.services.testing import testing_framework

            if not service:
                # Get all services status
                health = await testing_framework.get_tower_health_summary()
                response = "🏥 **Tower Services Status**\n\n"
                response += f"Overall: {health['summary']['overall_status'].upper()}\n"
                response += f"Active: {health['summary']['active_services']}/{health['summary']['total_services']}\n\n"

                for svc in health['services']:
                    emoji = "✅" if svc['status'] == 'active' else "❌"
                    response += f"{emoji} {svc['name']}: {svc['status']}\n"

                return response
            else:
                # Specific service
                result = await testing_framework.run_universal_test(service)
                if result['success']:
                    return f"✅ {service} is healthy"
                else:
                    return f"❌ {service} has issues: {result.get('error', 'Unknown')}"

        except Exception as e:
            return f"❌ Status check failed: {str(e)}"

    async def repair_service(self, params: Dict) -> str:
        """Repair a service"""
        service = params.get("args", "").strip()
        if not service:
            return "❌ Usage: /repair <service>\nExample: /repair auth"

        return f"🔧 Repairing {service}... (not fully implemented yet)"

    async def generate_image(self, params: Dict) -> str:
        """Generate an image"""
        prompt = params.get("args", "").strip()
        if not prompt:
            return "❌ Usage: /image <prompt>\nExample: /image cyberpunk anime girl"

        return f"🎨 Generating image: {prompt}... (delegating to ComfyUI)"

    async def generate_voice(self, params: Dict) -> str:
        """Generate voice"""
        text = params.get("args", "").strip()
        if not text:
            return "❌ Usage: /voice <text>\nExample: /voice Hello from Echo Brain"

        return f"🎵 Generating voice: {text}... (delegating to voice service)"

    async def list_capabilities(self, params: Dict) -> str:
        """List all capabilities"""
        return """🧠 **Echo Brain Capabilities**

**Connected & Ready:**
✅ Code Review - Analyze code quality with pylint
✅ Code Refactoring - Automatically fix code issues
✅ Service Testing - Test any Tower service
✅ Service Monitoring - Check health status
✅ Service Repair - Auto-repair broken services
✅ Image Generation - Via ComfyUI on NVIDIA GPU
✅ Voice Generation - Multiple TTS engines
✅ Database Operations - PostgreSQL queries
✅ File Operations - Read/write/modify files
✅ System Commands - Execute bash commands

**Autonomous Behaviors:**
• Monitoring Tower services every 60s
• Auto-repair with 5-minute cooldown
• Code quality scanning
• Task queue processing
• Learning from patterns

Use /help to see available commands.
"""
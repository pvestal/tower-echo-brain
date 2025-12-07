#!/usr/bin/env python3
"""
Agent Development Endpoints for AI Assist
API endpoints for developing and managing Echo's agent capabilities
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from echo_agent_development import EchoAgentDeveloper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI router for agent development
agent_dev_router = APIRouter(
    prefix="/api/agent-development", tags=["agent-development"]
)

# Global agent developer instance
agent_developer = None


# Request models
class AgentSpecification(BaseModel):
    name: str
    type: str = "task_agent"
    description: str
    capabilities: List[str]
    requirements: List[str]
    tools: List[str]


class DevelopmentRequest(BaseModel):
    agent_spec: AgentSpecification
    test_immediately: bool = True
    deploy_immediately: bool = False


def initialize_agent_developer():
    """Initialize the agent developer"""
    global agent_developer
    if not agent_developer:
        agent_developer = EchoAgentDeveloper()
        logger.info("🤖 Agent Developer initialized")
    return agent_developer


@agent_dev_router.on_event("startup")
async def startup_agent_developer():
    """Initialize agent developer on startup"""
    initialize_agent_developer()


@agent_dev_router.get("/status")
async def get_agent_development_status():
    """Get the current status of the agent development system"""
    if not agent_developer:
        initialize_agent_developer()

    try:
        await agent_developer.initialize_agent_development()
        status = await agent_developer.get_development_status()
        return {
            "service": "Echo Agent Development System",
            "status": "operational",
            "capabilities": [
                "Agent Architecture Design",
                "Autonomous Agent Generation",
                "Tool Integration",
                "Capability Testing",
                "Agent Deployment",
            ],
            **status,
        }
    except Exception as e:
        logger.error(f"Error getting agent development status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting status: {str(e)}")


@agent_dev_router.get("/tools")
async def discover_available_tools():
    """Discover and list all available tools for agent development"""
    if not agent_developer:
        initialize_agent_developer()

    try:
        await agent_developer.discover_available_tools()
        return {
            "available_tools": agent_developer.available_tools,
            "total_tools": len(agent_developer.available_tools),
            "tool_categories": {
                "intelligence_systems": len(
                    [
                        t
                        for t in agent_developer.available_tools
                        if t.get("type") == "intelligence_system"
                    ]
                ),
                "tower_services": len(
                    [
                        t
                        for t in agent_developer.available_tools
                        if t.get("type") == "tower_service"
                    ]
                ),
                "system_tools": len(
                    [
                        t
                        for t in agent_developer.available_tools
                        if t.get("type") == "system_tool"
                    ]
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error discovering tools: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error discovering tools: {str(e)}"
        )


@agent_dev_router.get("/capabilities")
async def analyze_echo_capabilities():
    """Analyze Echo's current capabilities and performance"""
    if not agent_developer:
        initialize_agent_developer()

    try:
        await agent_developer.analyze_echo_capabilities()
        return {
            "echo_capabilities": agent_developer.agent_capabilities,
            "capability_analysis": {
                "total_tests": len(agent_developer.agent_capabilities),
                "performance_metrics": {
                    test_name: {
                        "expected_vs_actual": cap["expected_level"]
                        == cap["actual_level"],
                        "response_time": cap["response_time"],
                        "model_used": cap["model_used"],
                    }
                    for test_name, cap in agent_developer.agent_capabilities.items()
                },
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error analyzing capabilities: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error analyzing capabilities: {str(e)}"
        )


@agent_dev_router.post("/develop-agent")
async def develop_new_agent(
    development_request: DevelopmentRequest, background_tasks: BackgroundTasks
):
    """Develop a new autonomous agent based on specifications"""
    if not agent_developer:
        initialize_agent_developer()

    try:
        # Convert Pydantic model to dict
        agent_spec = development_request.agent_spec.dict()

        logger.info(f"🤖 Starting development of agent: {agent_spec['name']}")

        # Start development in background
        background_tasks.add_task(
            develop_agent_background,
            agent_spec,
            development_request.test_immediately,
            development_request.deploy_immediately,
        )

        return {
            "development_started": True,
            "agent_name": agent_spec["name"],
            "development_type": agent_spec["type"],
            "estimated_time": "2-5 minutes",
            "status_endpoint": f"/api/agent-development/sessions/{agent_spec['name']}",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error starting agent development: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error starting development: {str(e)}"
        )


async def develop_agent_background(
    agent_spec: Dict, test_immediately: bool, deploy_immediately: bool
):
    """Develop agent in background task"""
    try:
        # Initialize if needed
        if not agent_developer.available_tools:
            await agent_developer.initialize_agent_development()

        # Develop the agent
        development_session = await agent_developer.develop_autonomous_agent(agent_spec)

        logger.info(
            f"🤖 Agent development completed: {agent_spec['name']} - Status: {development_session['status']}"
        )

    except Exception as e:
        logger.error(f"Background agent development failed: {e}")


@agent_dev_router.get("/sessions")
async def get_development_sessions():
    """Get all agent development sessions"""
    if not agent_developer:
        initialize_agent_developer()

    try:
        return {
            "development_sessions": agent_developer.development_sessions,
            "total_sessions": len(agent_developer.development_sessions),
            "session_summary": {
                "in_progress": len(
                    [
                        s
                        for s in agent_developer.development_sessions
                        if s["status"] == "in_progress"
                    ]
                ),
                "completed": len(
                    [
                        s
                        for s in agent_developer.development_sessions
                        if s["status"] == "completed"
                    ]
                ),
                "failed": len(
                    [
                        s
                        for s in agent_developer.development_sessions
                        if s["status"] == "failed"
                    ]
                ),
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting development sessions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting sessions: {str(e)}")


@agent_dev_router.get("/sessions/{agent_name}")
async def get_development_session(agent_name: str):
    """Get a specific development session by agent name"""
    if not agent_developer:
        initialize_agent_developer()

    try:
        # Find session by agent name
        session = next(
            (
                s
                for s in agent_developer.development_sessions
                if s.get("agent_name") == agent_name
            ),
            None,
        )

        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Development session for {agent_name} not found",
            )

        return {
            "session": session,
            "agent_name": agent_name,
            "development_progress": len(session.get("steps", [])),
            "current_status": session.get("status", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting development session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error getting session: {str(e)}")


@agent_dev_router.post("/demo")
async def run_agent_development_demo():
    """Run a comprehensive demonstration of agent development"""
    if not agent_developer:
        initialize_agent_developer()

    try:
        logger.info("🚀 Starting Agent Development Demo")
        demo_results = await agent_developer.run_agent_development_demo()

        return {
            "demo_status": "completed",
            "demo_results": demo_results,
            "agents_created": demo_results.get("agents_developed", []),
            "development_time": "5-10 minutes",
            "next_steps": [
                "Review generated agent code",
                "Test agent capabilities",
                "Deploy agents to Echo ecosystem",
                "Monitor agent performance",
            ],
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error running demo: {e}")
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


@agent_dev_router.get("/templates")
async def get_agent_templates():
    """Get available agent templates for development"""
    templates = {
        "task_agent": {
            "name": "Task Execution Agent",
            "description": "Specialized in executing specific tasks using available tools",
            "default_capabilities": ["tool_usage", "task_breakdown", "error_handling"],
            "example_requirements": [
                "execute complex tasks",
                "coordinate multiple tools",
                "handle failures gracefully",
            ],
            "recommended_tools": ["echo_brain", "system_tools", "tower_services"],
        },
        "research_agent": {
            "name": "Research Agent",
            "description": "Specialized in information gathering and research tasks",
            "default_capabilities": ["information_gathering", "analysis", "synthesis"],
            "example_requirements": [
                "search for information",
                "analyze findings",
                "generate reports",
            ],
            "recommended_tools": ["web_search", "knowledge_base", "echo_brain"],
        },
        "coordination_agent": {
            "name": "Coordination Agent",
            "description": "Specialized in coordinating multiple agents and complex workflows",
            "default_capabilities": [
                "agent_management",
                "workflow_orchestration",
                "conflict_resolution",
            ],
            "example_requirements": [
                "manage multiple agents",
                "orchestrate workflows",
                "resolve conflicts",
            ],
            "recommended_tools": ["echo_brain", "agent_manager", "workflow_engine"],
        },
        "analysis_agent": {
            "name": "Analysis Agent",
            "description": "Specialized in data analysis and pattern recognition",
            "default_capabilities": [
                "data_processing",
                "pattern_recognition",
                "insight_generation",
            ],
            "example_requirements": [
                "analyze data patterns",
                "generate insights",
                "create visualizations",
            ],
            "recommended_tools": [
                "data_processing",
                "echo_brain",
                "visualization_tools",
            ],
        },
        "communication_agent": {
            "name": "Communication Agent",
            "description": "Specialized in communication and interaction management",
            "default_capabilities": [
                "message_processing",
                "response_generation",
                "context_management",
            ],
            "example_requirements": [
                "handle user interactions",
                "generate responses",
                "maintain context",
            ],
            "recommended_tools": ["echo_brain", "nlp_tools", "communication_channels"],
        },
    }

    return {
        "available_templates": templates,
        "total_templates": len(templates),
        "template_categories": list(templates.keys()),
        "usage_guide": {
            "step_1": "Choose a template type",
            "step_2": "Customize capabilities and requirements",
            "step_3": "Specify tools needed",
            "step_4": "Submit development request",
            "step_5": "Monitor development progress",
        },
        "timestamp": datetime.now().isoformat(),
    }


@agent_dev_router.post("/quick-agent")
async def create_quick_agent(
    background_tasks: BackgroundTasks,
    name: str,
    template: str = "task_agent",
    description: str = "",
):
    """Quickly create an agent using a template"""
    if not agent_developer:
        initialize_agent_developer()

    # Get template details
    templates_response = await get_agent_templates()
    templates = templates_response["available_templates"]

    if template not in templates:
        raise HTTPException(
            status_code=400, detail=f"Template '{template}' not found")

    template_config = templates[template]

    # Create agent specification from template
    agent_spec = {
        "name": name,
        "type": template,
        "description": description or template_config["description"],
        "capabilities": template_config["default_capabilities"],
        "requirements": template_config["example_requirements"],
        "tools": template_config["recommended_tools"],
    }

    try:
        # Start development in background
        background_tasks.add_task(
            develop_agent_background,
            agent_spec,
            True,  # test_immediately
            False,  # deploy_immediately
        )

        return {
            "quick_agent_creation": True,
            "agent_name": name,
            "template_used": template,
            "agent_specification": agent_spec,
            "development_started": True,
            "estimated_completion": "3-5 minutes",
            "status_endpoint": f"/api/agent-development/sessions/{name}",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error creating quick agent: {e}")
        raise HTTPException(
            status_code=500, detail=f"Quick agent creation failed: {str(e)}"
        )


@agent_dev_router.get("/health")
async def agent_development_health():
    """Health check for agent development system"""
    return {
        "service": "Echo Agent Development System",
        "status": "healthy",
        "capabilities": [
            "Autonomous Agent Generation",
            "Tool Discovery & Integration",
            "Capability Testing",
            "Template-based Development",
            "Performance Analysis",
        ],
        "developer_initialized": agent_developer is not None,
        "development_workspace": "/opt/tower-echo-brain/agent_development",
        "timestamp": datetime.now().isoformat(),
    }


# Export router
__all__ = ["agent_dev_router", "initialize_agent_developer"]

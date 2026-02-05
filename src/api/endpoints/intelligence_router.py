"""
Echo Brain Intelligence API - New Intelligence Layer
Real intelligence with code understanding, system monitoring, and action execution
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import json
from datetime import datetime

# Import new intelligence components
from src.intelligence.reasoner import get_reasoning_engine
from src.intelligence.code_index import get_code_intelligence
from src.intelligence.system_model import get_system_model
from src.intelligence.procedures import get_procedure_library
from src.intelligence.executor import get_action_executor
from src.intelligence.schemas import QueryRequest, ActionRequest, DiagnoseRequest

router = APIRouter(prefix="/intelligence", tags=["intelligence"])
logger = logging.getLogger(__name__)

# API Models
class QueryResponse(BaseModel):
    query: str
    query_type: str
    response: str
    actions_taken: List[Dict[str, Any]]
    confidence: float
    sources: List[str]
    execution_time_ms: int

class ActionResponse(BaseModel):
    action: str
    success: bool
    result: Dict[str, Any]
    requires_confirmation: Optional[bool] = None

class DiagnoseResponse(BaseModel):
    issue: str
    findings: List[str]
    root_cause: Optional[str]
    recommendations: List[str]
    severity: str
    estimated_fix_time: Optional[int]

# Intelligence endpoints
@router.post("/query", response_model=QueryResponse)
async def intelligent_query(request: QueryRequest):
    """Main entry point - routes to appropriate intelligence"""
    try:
        reasoner = get_reasoning_engine()

        response = await reasoner.process(
            request.query,
            allow_actions=request.allow_actions,
            context=request.context
        )

        return QueryResponse(
            query=response.query,
            query_type=response.query_type.value,
            response=response.response,
            actions_taken=response.actions_taken,
            confidence=response.confidence,
            sources=response.sources,
            execution_time_ms=response.execution_time_ms
        )

    except Exception as e:
        logger.error(f"Intelligence query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute", response_model=ActionResponse)
async def execute_action(request: ActionRequest):
    """Execute an action with confirmation"""
    try:
        reasoner = get_reasoning_engine()

        # Create a query for the action
        query = f"execute {request.action}"

        response = await reasoner.process(
            query,
            allow_actions=True,
            context=request.parameters
        )

        # Extract action results
        action_result = {
            'success': len(response.actions_taken) > 0 and all(
                a.get('success', False) for a in response.actions_taken
            ),
            'actions_taken': response.actions_taken,
            'response': response.response
        }

        return ActionResponse(
            action=request.action,
            success=action_result['success'],
            result=action_result,
            requires_confirmation=request.confirm_dangerous
        )

    except Exception as e:
        logger.error(f"Action execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/explain/{path:path}")
async def explain_code(path: str):
    """Explain what a code file/function does"""
    try:
        reasoner = get_reasoning_engine()

        explanation = await reasoner.explain_code(f"explain {path}")

        return {
            "path": path,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Code explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/service/{name}")
async def get_service_info(name: str):
    """Get detailed info about a service"""
    try:
        system_model = get_system_model()

        status = await system_model.get_service_status(name)
        dependencies = await system_model.get_service_dependencies(name)
        config = await system_model.get_service_config(name)

        return {
            "service": name,
            "status": status.dict(),
            "dependencies": dependencies,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Service info failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose_issue(request: DiagnoseRequest):
    """Run diagnostic procedure"""
    try:
        reasoner = get_reasoning_engine()

        diagnosis = await reasoner.diagnose(request.issue)

        return DiagnoseResponse(
            issue=diagnosis.issue,
            findings=diagnosis.findings,
            root_cause=diagnosis.root_cause,
            recommendations=diagnosis.recommendations,
            severity=diagnosis.severity,
            estimated_fix_time=diagnosis.estimated_fix_time
        )

    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/procedures")
async def list_procedures():
    """List available procedures"""
    try:
        procedures = get_procedure_library()

        procedure_list = await procedures.list_procedures()

        return {
            "procedures": procedure_list,
            "total_count": len(procedure_list),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Procedure listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/procedures/{name}/execute")
async def run_procedure(name: str, context: Dict[str, Any] = None):
    """Execute a named procedure"""
    try:
        procedures = get_procedure_library()

        # Find the procedure
        procedure = await procedures.find_procedure(name)

        if not procedure:
            raise HTTPException(status_code=404, detail=f"Procedure '{name}' not found")

        # Execute the procedure
        result = await procedures.execute_procedure(
            procedure,
            context or {},
            allow_dangerous=False  # Require explicit confirmation for dangerous operations
        )

        return {
            "procedure": name,
            "execution_result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Procedure execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Code intelligence endpoints
@router.post("/code/index")
async def index_code(paths: List[str] = None):
    """Index/re-index codebase"""
    try:
        code_intel = get_code_intelligence()

        if not paths:
            # Default paths to index
            paths = [
                "/opt/tower-echo-brain/src",
                "/opt/tower-auth/src",
                "/opt/tower-kb/src"
            ]

        result = await code_intel.index_codebase(paths)

        return {
            "indexing_result": result,
            "paths_processed": paths,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Code indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/code/search")
async def search_code(query: str, type: str = None):
    """Search code symbols"""
    try:
        code_intel = get_code_intelligence()

        symbols = await code_intel.search_symbols(query, type)

        return {
            "query": query,
            "type_filter": type,
            "symbols": symbols,
            "count": len(symbols),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Code search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/code/dependencies/{path:path}")
async def get_dependencies(path: str):
    """Get file dependencies"""
    try:
        code_intel = get_code_intelligence()

        # Ensure path starts with /
        if not path.startswith('/'):
            path = '/' + path

        deps = await code_intel.get_dependencies(path)

        return {
            "file_path": path,
            "dependencies": deps.dict(),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Dependency analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System model endpoints
@router.get("/system/services")
async def list_services():
    """List all known services"""
    try:
        system_model = get_system_model()

        services = await system_model.discover_services()

        return {
            "services": [s.dict() for s in services],
            "count": len(services),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Service listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/topology")
async def get_topology():
    """Get service dependency graph"""
    try:
        system_model = get_system_model()

        network_map = await system_model.get_network_map()

        return {
            "topology": network_map.dict(),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Topology mapping failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system/scan")
async def scan_system():
    """Re-scan system for services"""
    try:
        system_model = get_system_model()

        services = await system_model.discover_services()

        return {
            "scan_result": "System scan completed",
            "services_found": len(services),
            "services": [s.dict() for s in services[:10]],  # First 10 services
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"System scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_intelligence_status():
    """Get status of all intelligence components"""
    try:
        status = {
            "components": {
                "reasoning_engine": "Available",
                "code_intelligence": "Available",
                "system_model": "Available",
                "procedure_library": "Available",
                "action_executor": "Available"
            },
            "database_connectivity": True,
            "timestamp": datetime.now().isoformat()
        }

        # Try to get some stats
        try:
            procedures = get_procedure_library()
            procedure_list = await procedures.list_procedures()
            status["procedure_count"] = len(procedure_list)
        except:
            status["procedure_count"] = 0

        try:
            system_model = get_system_model()
            services = await system_model.discover_services()
            status["services_monitored"] = len(services)
        except:
            status["services_monitored"] = 0

        return status

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "components": {
                "reasoning_engine": "Error",
                "code_intelligence": "Error",
                "system_model": "Error",
                "procedure_library": "Error",
                "action_executor": "Error"
            },
            "database_connectivity": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
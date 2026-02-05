"""
Setup script for initializing the intelligence layer
"""

import asyncio
import logging
import sys
import os
from typing import List, Dict, Any

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.intelligence.procedures import get_procedure_library
from src.intelligence.code_index import get_code_intelligence
from src.intelligence.system_model import get_system_model

logger = logging.getLogger(__name__)


async def initialize_intelligence_layer() -> Dict[str, Any]:
    """Initialize all intelligence components"""
    results = {
        'procedures_initialized': False,
        'code_indexed': False,
        'system_scanned': False,
        'errors': []
    }

    try:
        logger.info("Initializing intelligence layer...")

        # 1. Initialize procedures
        logger.info("Initializing procedure library...")
        procedures = get_procedure_library()
        await procedures.initialize_procedures()
        results['procedures_initialized'] = True
        logger.info("Procedure library initialized")

        # 2. Index Tower codebase
        logger.info("Indexing Tower codebase...")
        code_intel = get_code_intelligence()

        paths_to_index = [
            "/opt/tower-echo-brain/src",
        ]

        # Check if other Tower services exist
        import os
        optional_paths = [
            "/opt/tower-auth/src",
            "/opt/tower-kb/src",
            "/opt/tower-dashboard/src",
            "/opt/tower-apple-music/src"
        ]

        for path in optional_paths:
            if os.path.exists(path):
                paths_to_index.append(path)

        indexing_result = await code_intel.index_codebase(paths_to_index)
        results['code_indexed'] = True
        results['indexing_result'] = indexing_result
        logger.info(f"Code indexing completed: {indexing_result}")

        # 3. Scan system services
        logger.info("Scanning system services...")
        system_model = get_system_model()
        services = await system_model.discover_services()
        results['system_scanned'] = True
        results['services_found'] = len(services)
        logger.info(f"System scan completed: {len(services)} services found")

        logger.info("Intelligence layer initialization completed successfully")

    except Exception as e:
        logger.error(f"Intelligence layer initialization failed: {e}")
        results['errors'].append(str(e))

    return results


async def test_intelligence_components() -> Dict[str, Any]:
    """Test all intelligence components"""
    results = {
        'components_tested': 0,
        'components_passed': 0,
        'test_results': {},
        'errors': []
    }

    # Test CodeIntelligence
    try:
        logger.info("Testing CodeIntelligence...")
        code_intel = get_code_intelligence()

        # Test symbol search
        symbols = await code_intel.search_symbols("process", "function")
        results['test_results']['code_intelligence'] = {
            'status': 'passed',
            'symbols_found': len(symbols),
            'test': 'search_symbols'
        }
        results['components_passed'] += 1

    except Exception as e:
        logger.error(f"CodeIntelligence test failed: {e}")
        results['test_results']['code_intelligence'] = {
            'status': 'failed',
            'error': str(e)
        }
        results['errors'].append(f"CodeIntelligence: {e}")

    results['components_tested'] += 1

    # Test SystemModel
    try:
        logger.info("Testing SystemModel...")
        system_model = get_system_model()

        # Test service status
        status = await system_model.get_service_status('tower-echo-brain')
        results['test_results']['system_model'] = {
            'status': 'passed',
            'echo_brain_status': status.status,
            'test': 'get_service_status'
        }
        results['components_passed'] += 1

    except Exception as e:
        logger.error(f"SystemModel test failed: {e}")
        results['test_results']['system_model'] = {
            'status': 'failed',
            'error': str(e)
        }
        results['errors'].append(f"SystemModel: {e}")

    results['components_tested'] += 1

    # Test ProcedureLibrary
    try:
        logger.info("Testing ProcedureLibrary...")
        procedures = get_procedure_library()

        # Test procedure listing
        procedure_list = await procedures.list_procedures()
        results['test_results']['procedure_library'] = {
            'status': 'passed',
            'procedures_available': len(procedure_list),
            'test': 'list_procedures'
        }
        results['components_passed'] += 1

    except Exception as e:
        logger.error(f"ProcedureLibrary test failed: {e}")
        results['test_results']['procedure_library'] = {
            'status': 'failed',
            'error': str(e)
        }
        results['errors'].append(f"ProcedureLibrary: {e}")

    results['components_tested'] += 1

    # Test ReasoningEngine
    try:
        logger.info("Testing ReasoningEngine...")
        from .reasoner import get_reasoning_engine
        reasoner = get_reasoning_engine()

        # Test basic query processing
        response = await reasoner.process("What services are running?", allow_actions=False)
        results['test_results']['reasoning_engine'] = {
            'status': 'passed',
            'query_type': response.query_type.value,
            'confidence': response.confidence,
            'test': 'process_query'
        }
        results['components_passed'] += 1

    except Exception as e:
        logger.error(f"ReasoningEngine test failed: {e}")
        results['test_results']['reasoning_engine'] = {
            'status': 'failed',
            'error': str(e)
        }
        results['errors'].append(f"ReasoningEngine: {e}")

    results['components_tested'] += 1

    logger.info(f"Intelligence testing completed: {results['components_passed']}/{results['components_tested']} passed")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def main():
        print("Echo Brain Intelligence Layer Setup")
        print("=" * 50)

        # Initialize
        print("\n1. Initializing intelligence components...")
        init_results = await initialize_intelligence_layer()

        if init_results['procedures_initialized']:
            print("✅ Procedures initialized")
        else:
            print("❌ Procedure initialization failed")

        if init_results['code_indexed']:
            print(f"✅ Code indexed: {init_results.get('indexing_result', {}).get('files_processed', 0)} files")
        else:
            print("❌ Code indexing failed")

        if init_results['system_scanned']:
            print(f"✅ System scanned: {init_results.get('services_found', 0)} services")
        else:
            print("❌ System scan failed")

        # Test
        print("\n2. Testing intelligence components...")
        test_results = await test_intelligence_components()

        for component, result in test_results['test_results'].items():
            if result['status'] == 'passed':
                print(f"✅ {component.title()}")
            else:
                print(f"❌ {component.title()}: {result.get('error', 'Unknown error')}")

        # Summary
        print(f"\n3. Summary:")
        print(f"Components tested: {test_results['components_tested']}")
        print(f"Components passed: {test_results['components_passed']}")
        print(f"Success rate: {test_results['components_passed']/test_results['components_tested']*100:.1f}%")

        if test_results['errors']:
            print("\nErrors encountered:")
            for error in test_results['errors']:
                print(f"  - {error}")

    asyncio.run(main())
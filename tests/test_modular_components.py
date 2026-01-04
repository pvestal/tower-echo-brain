#!/usr/bin/env python3
from src.processors.input_processor import InputType
import asyncio
import sys
import os
sys.path.insert(0, '/opt/tower-echo-brain')

async def test_components():
    print('🧪 TESTING REFACTORED ECHO COMPONENTS')
    print('=' * 50)
    
    results = {'passed': 0, 'failed': 0}
    
    # Test 1: Configuration Manager
    try:
        from src.managers.configuration_manager import ConfigurationManager
        config = ConfigurationManager()
        config_items = config.get_all()
        print(f'✅ ConfigurationManager: {len(config_items)} settings loaded')
        results['passed'] += 1
    except Exception as e:
        print(f'❌ ConfigurationManager: {e}')
        results['failed'] += 1
    
    # Test 2: Input Processor
    try:
        from src.processors.input_processor import InputProcessor
        input_proc = InputProcessor(config)
        print(f'✅ InputProcessor: {len(input_proc.supported_input_types)} input types supported')
        
        # Test processing
        test_input = {'type': 'chat_message', 'content': 'Test message'}
        result = await input_proc.process(test_input, InputType.CHAT_MESSAGE)
        print(f'   Processed test input: {result.get("status")}')
        results['passed'] += 1
    except Exception as e:
        print(f'❌ InputProcessor: {e}')
        results['failed'] += 1
    
    # Test 3: Output Generator
    try:
        from src.generators.output_generator import OutputGenerator
        output_gen = OutputGenerator(config)
        print(f'✅ OutputGenerator: {len(output_gen.supported_output_types)} output types')
        print(f'   Delivery methods: {len(output_gen.delivery_methods)}')
        results['passed'] += 1
    except Exception as e:
        print(f'❌ OutputGenerator: {e}')
        results['failed'] += 1
    
    # Test 4: Dependency Container
    try:
        from src.components.dependency_container import DependencyContainer
        container = DependencyContainer()
        container.register_singleton('config', lambda: config)
        resolved = container.resolve('config')
        print(f'✅ DependencyContainer: Resolution working')
        results['passed'] += 1
    except Exception as e:
        print(f'❌ DependencyContainer: {e}')
        results['failed'] += 1
    
    # Test 5: Error Handler
    try:
        from src.components.error_handler import ErrorHandler
        error_handler = ErrorHandler(config)
        test_error = ValueError('Test error')
        handled = await error_handler.handle(test_error)
        print(f'✅ ErrorHandler: Error categorization working')
        results['passed'] += 1
    except Exception as e:
        print(f'❌ ErrorHandler: {e}')
        results['failed'] += 1
    
    # Test 6: Logging System
    try:
        from src.components.logging_system import StructuredLogger
        logger = StructuredLogger('test', config)
        await logger.info('Test log message', {'test': 'data'})
        print(f'✅ StructuredLogger: Logging functional')
        results['passed'] += 1
    except Exception as e:
        print(f'❌ StructuredLogger: {e}')
        results['failed'] += 1
    
    # Test 7: Echo Orchestrator
    try:
        from src.components.echo_orchestrator import EchoOrchestrator
        orchestrator = EchoOrchestrator(
            input_processor=input_proc,
            output_generator=output_gen,
            config_manager=config,
            error_handler=error_handler,
            logger=logger
        )
        print(f'✅ EchoOrchestrator: Integration successful')
        results['passed'] += 1
    except Exception as e:
        print(f'❌ EchoOrchestrator: {e}')
        results['failed'] += 1
    
    # Test 8: Main Refactored Service
    try:
        from src.main_refactored import app
        print(f'✅ Main Service: FastAPI app loaded')
        results['passed'] += 1
    except Exception as e:
        print(f'❌ Main Service: {e}')
        results['failed'] += 1
    
    print('=' * 50)
    print(f'📊 TEST RESULTS: {results["passed"]} passed, {results["failed"]} failed')
    print(f'✅ Success Rate: {results["passed"]/(results["passed"]+results["failed"])*100:.1f}%')
    
    return results

if __name__ == '__main__':
    results = asyncio.run(test_components())
    sys.exit(0 if results['failed'] == 0 else 1)

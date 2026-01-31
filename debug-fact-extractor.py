#!/usr/bin/env python3
import asyncio
import sys
import os
sys.path.insert(0, '/opt/tower-echo-brain/echo-brain-context-pipeline')

async def main():
    # Import the module components
    import src.ingestion.fact_extractor as module
    
    print("=== Debugging Fact Extractor ===")
    print(f"Module: {module.__file__}")
    
    # Check what's in the module
    print("\nModule contents:")
    for item in dir(module):
        if not item.startswith('_'):
            print(f"  {item}")
    
    # Try to run the main function
    if hasattr(module, 'run_full_extraction'):
        print("\nFound run_full_extraction function")
        try:
            dsn = "postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain"
            print(f"Running with DSN: {dsn[:50]}...")
            await module.run_full_extraction(dsn)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo run_full_extraction function found")

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import sys
sys.path.insert(0, '/opt/tower-echo-brain/echo-brain-context-pipeline')

async def test():
    try:
        from src.ingestion.fact_extractor import FactExtractor
        print("✅ FactExtractor imported successfully")
        
        # Create extractor instance
        extractor = FactExtractor('postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain')
        print("✅ FactExtractor instance created")
        
        # Test with just 2 items
        print("Processing 2 items as test...")
        result = await extractor.extract_all_pending(limit=2)
        print(f"✅ Test result: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())

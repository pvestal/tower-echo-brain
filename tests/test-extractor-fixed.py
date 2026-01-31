import asyncio
import sys
sys.path.insert(0, '/opt/tower-echo-brain/echo-brain-context-pipeline')

async def test():
    try:
        from src.ingestion.fact_extractor import FactExtractor
        print("✅ FactExtractor imported successfully")
        
        # Try different ways to instantiate
        print("Attempt 1: With postgres_dsn keyword argument...")
        try:
            extractor = FactExtractor(postgres_dsn='postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain')
            print("✅ FactExtractor instance created (keyword argument)")
        except Exception as e1:
            print(f"❌ Attempt 1 failed: {e1}")
            
            print("Attempt 2: With positional argument...")
            try:
                extractor = FactExtractor('postgresql://patrick:RP78eIrW7cI2jYvL5akt1yurE@localhost/echo_brain')
                print("✅ FactExtractor instance created (positional argument)")
            except Exception as e2:
                print(f"❌ Attempt 2 failed: {e2}")
                return
        
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

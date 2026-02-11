import asyncio
from src.autonomous.workers.fact_extraction_worker import FactExtractionWorker

async def test():
    worker = FactExtractionWorker()
    # Test with a known piece of text
    test_text = "Tower server has 96GB DDR6 RAM and an NVIDIA RTX 3060 12GB GPU."
    print(f"Testing extraction on: {test_text}")
    # You'll need to check the actual extraction method name

asyncio.run(test())

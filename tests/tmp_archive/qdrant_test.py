#!/usr/bin/env python3
# Run this as a Python script, not a summary
import requests

# Collection info
info = requests.get("http://localhost:6333/collections/echo_memory").json()
result = info.get("result", {})
config = result.get("config", {})
vectors_config = config.get("params", {}).get("vectors", {})
points_count = result.get("points_count", 0)
vectors_count = result.get("vectors_count", 0)

print(f"Collection: echo_memory")
print(f"Vector dimensions: {vectors_config.get('size', 'UNKNOWN')}")
print(f"Points count: {points_count}")
print(f"Vectors count: {vectors_count}")

if points_count == 0:
    print("⚠️  EMPTY COLLECTION — Echo Brain has no memory at all")
else:
    # Check what types of vectors exist
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = QdrantClient(host="localhost", port=6333)

    for vtype in ["kb_article", "conversation", "code", "documentation", "facts"]:
        try:
            results = client.scroll(
                collection_name="echo_memory",
                scroll_filter=Filter(must=[
                    FieldCondition(key="type", match=MatchValue(value=vtype))
                ]),
                limit=1
            )
            count_check = client.count(
                collection_name="echo_memory",
                count_filter=Filter(must=[
                    FieldCondition(key="type", match=MatchValue(value=vtype))
                ])
            )
            sample = results[0][0].payload if results[0] else None
            content_preview = ""
            if sample:
                content_preview = sample.get("content", sample.get("text", "NO CONTENT FIELD"))[:150]
            print(f"  {vtype}: {count_check.count} vectors | Sample: {content_preview}")
        except Exception as e:
            print(f"  {vtype}: ERROR - {e}")

    # Check for untyped legacy vectors
    try:
        all_count = client.count(collection_name="echo_memory")
        typed_total = sum(
            client.count(
                collection_name="echo_memory",
                count_filter=Filter(must=[
                    FieldCondition(key="type", match=MatchValue(value=t))
                ])
            ).count
            for t in ["kb_article", "conversation", "code", "documentation", "facts"]
        )
        untyped = all_count.count - typed_total
        if untyped > 0:
            print(f"\n  ⚠️  UNTYPED/LEGACY vectors: {untyped} — these are from old ingestion and may be wrong dimensions or corrupted")
    except Exception as e:
        print(f"  Legacy check error: {e}")
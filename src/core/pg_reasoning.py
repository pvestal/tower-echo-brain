"""Simple reasoning using PostgreSQL search + Ollama LLM"""
import psycopg2
import requests
import time
from typing import List, Dict

DB = {"host": "localhost", "database": "echo_brain", "user": "patrick", "password": "RP78eIrW7cI2jYvL5akt1yurE"}
OLLAMA_URL = "http://localhost:11434"

def search_pg(query: str, limit: int = 5) -> List[Dict]:
    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT conversation_id, role, content
        FROM claude_conversations
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
           OR content ILIKE %s
        ORDER BY ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) DESC
        LIMIT %s
    """, (query, f'%{query}%', query, limit))
    results = [{"conv": r[0], "role": r[1], "content": r[2]} for r in cur.fetchall()]
    cur.close()
    conn.close()
    return results

def ask_with_pg(question: str) -> Dict:
    start = time.time()
    memories = search_pg(question, limit=5)
    search_time = (time.time() - start) * 1000
    
    context = "\n\n".join([f"[{m['role']}]: {m['content'][:1500]}" for m in memories[:5]])
    
    prompt = f"""You are Echo Brain, an AI assistant with memory of past conversations.

Based on these relevant past conversations:
{context}

Question: {question}

Provide a helpful answer based on the context. If no relevant context exists, say so clearly."""

    llm_start = time.time()
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": "mistral:7b",
        "prompt": prompt,
        "stream": False
    }, timeout=120)
    llm_time = (time.time() - llm_start) * 1000
    
    answer = resp.json().get("response", "Error: No response from LLM")
    
    return {
        "query": question,
        "answer": answer,
        "confidence": 0.85 if memories else 0.3,
        "memories_used": len(memories),
        "sources": [m["conv"] for m in memories],
        "reasoning_time_ms": int(search_time + llm_time),
        "model_used": "mistral:7b",
        "search_time_ms": int(search_time),
        "llm_time_ms": int(llm_time)
    }

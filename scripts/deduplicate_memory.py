#!/usr/bin/env python3
"""
Deduplicate Echo Brain memory and vector data
"""
import psycopg2
import os
import hashlib
from typing import List, Dict, Any

class MemoryDeduplicator:
    def __init__(self):
        self.db_config = {
            "dbname": os.getenv("DB_NAME", "echo_brain"),
            "user": os.getenv("DB_USER", "echo"),
            "password": os.getenv("DB_PASSWORD", "echo_secure_password_123"),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432")
        }
    
    def find_duplicate_conversations(self) -> List[Dict[str, Any]]:
        """Find duplicate conversations by content hash"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Create temporary table with content hashes
        cursor.execute("""
            WITH content_hashes AS (
                SELECT 
                    id,
                    title,
                    content,
                    MD5(content) as content_hash,
                    created_at
                FROM conversations
                WHERE content IS NOT NULL AND content != ''
            )
            SELECT 
                content_hash,
                COUNT(*) as duplicate_count,
                ARRAY_AGG(id ORDER BY created_at) as ids,
                MIN(created_at) as first_created,
                MAX(created_at) as last_created
            FROM content_hashes
            GROUP BY content_hash
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC
        """)
        
        duplicates = []
        for row in cursor.fetchall():
            duplicates.append({
                "content_hash": row[0],
                "count": row[1],
                "ids": row[2],
                "first_created": row[3],
                "last_created": row[4]
            })
        
        cursor.close()
        conn.close()
        return duplicates
    
    def remove_duplicate_conversations(self, keep_oldest: bool = True):
        """Remove duplicate conversations, keeping the oldest or newest"""
        duplicates = self.find_duplicate_conversations()
        
        if not duplicates:
            print("No duplicate conversations found.")
            return
        
        print(f"Found {len(duplicates)} sets of duplicates")
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        total_removed = 0
        for dup in duplicates:
            # Keep the first ID (oldest), delete the rest
            keep_id = dup["ids"][0] if keep_oldest else dup["ids"][-1]
            delete_ids = [str(id) for id in dup["ids"] if id != keep_id]
            
            if delete_ids:
                delete_query = f"""
                    DELETE FROM conversations 
                    WHERE id IN ({','.join(delete_ids)})
                """
                cursor.execute(delete_query)
                total_removed += len(delete_ids)
                print(f"Removed {len(delete_ids)} duplicates for hash {dup['content_hash'][:16]}...")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ Removed {total_removed} duplicate conversations")
    
    def check_vector_duplicates(self):
        """Check for potential vector duplicates (by content similarity)"""
        # This would require Qdrant integration for semantic deduplication
        print("Vector deduplication requires Qdrant API integration")
        print("To implement: compare embedding similarity scores")

if __name__ == "__main__":
    dedup = MemoryDeduplicator()
    
    print("=== Echo Brain Memory Deduplication ===")
    print()
    
    # Find and report duplicates
    duplicates = dedup.find_duplicate_conversations()
    if duplicates:
        print(f"Found {len(duplicates)} duplicate conversation sets:")
        for dup in duplicates[:5]:  # Show first 5
            print(f"  - Hash: {dup['content_hash'][:16]}..., Count: {dup['count']}")
        
        response = input("\nRemove duplicates? (y/N): ")
        if response.lower() == 'y':
            dedup.remove_duplicate_conversations()
    else:
        print("✅ No duplicate conversations found.")

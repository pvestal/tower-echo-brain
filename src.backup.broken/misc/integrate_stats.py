#!/usr/bin/env python3
"""
Integrate stats tracking into existing Echo Brain
"""

def integrate_stats():
    # Read the original file
    with open("echo_working.py.backup_before_stats", "r") as f:
        content = f.read()
    
    # Add imports
    content = content.replace(
        "import requests",
        "import requests\nimport sqlite3"
    )
    
    content = content.replace(
        "from datetime import datetime",
        "from datetime import datetime, timedelta"
    )
    
    content = content.replace(
        "from fastapi import FastAPI, HTTPException",
        "from fastapi import FastAPI, HTTPException, Path"
    )
    
    # Add database functions after story_states
    db_functions = """
# Database initialization
DB_PATH = "/opt/tower-echo-brain/data/user_stats.db"

def init_database():
    \"\"\"Initialize SQLite database for user statistics\"\"\"
    import os
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create user_stats table
    cursor.execute(\"\"\"\n        CREATE TABLE IF NOT EXISTS user_stats (\n            user_id TEXT PRIMARY KEY,\n            total_generations INTEGER DEFAULT 0,\n            successful_generations INTEGER DEFAULT 0,\n            failed_generations INTEGER DEFAULT 0,\n            total_errors INTEGER DEFAULT 0,\n            last_generation_time TEXT,\n            first_seen TEXT,\n            last_active TEXT,\n            total_tokens_used INTEGER DEFAULT 0\n        )\n    \"\"\")
    
    # Create generation_log table for detailed tracking
    cursor.execute(\"\"\"\n        CREATE TABLE IF NOT EXISTS generation_log (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            user_id TEXT,\n            session_id TEXT,\n            timestamp TEXT,\n            prompt TEXT,\n            success BOOLEAN,\n            error_message TEXT,\n            image_path TEXT,\n            tokens_used INTEGER DEFAULT 0,\n            processing_time REAL,\n            FOREIGN KEY (user_id) REFERENCES user_stats (user_id)\n        )\n    \"\"\")
    
    conn.commit()
    conn.close()
    print("✅ Database initialized with user statistics tables")

def get_or_create_user_stats(user_id: str):
    \"\"\"Get or create user statistics record\"\"\"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
    user = cursor.fetchone()
    
    if not user:
        # Create new user record
        now = datetime.now().isoformat()
        cursor.execute(\"\"\"\n            INSERT INTO user_stats \n            (user_id, total_generations, successful_generations, failed_generations, \n             total_errors, first_seen, last_active, total_tokens_used)\n            VALUES (?, 0, 0, 0, 0, ?, ?, 0)\n        \"\"\", (user_id, now, now))
        conn.commit()
        print(f"📊 Created new user stats for: {user_id}")
    
    conn.close()

def update_user_activity(user_id: str):
    \"\"\"Update user s last active timestamp\"\"\"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(\"\"\"\n        UPDATE user_stats \n        SET last_active = ? \n        WHERE user_id = ?\n    \"\"\", (datetime.now().isoformat(), user_id))
    
    conn.commit()
    conn.close()

def log_generation_attempt(user_id: str, session_id: str, prompt: str, 
                          success: bool, error_message: str = None, 
                          image_path: str = None, processing_time: float = 0.0):
    \"\"\"Log generation attempt with full details\"\"\"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Estimate tokens (rough calculation: 1 token ≈ 4 characters)
    tokens_used = len(prompt) // 4 if prompt else 0
    
    # Log to generation_log
    cursor.execute(\"\"\"\n        INSERT INTO generation_log \n        (user_id, session_id, timestamp, prompt, success, error_message, \n         image_path, tokens_used, processing_time)\n        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)\n    \"\"\", (user_id, session_id, datetime.now().isoformat(), prompt, \n          success, error_message, image_path, tokens_used, processing_time))
    
    # Update user stats
    if success:
        cursor.execute(\"\"\"\n            UPDATE user_stats \n            SET total_generations = total_generations + 1,\n                successful_generations = successful_generations + 1,\n                last_generation_time = ?,\n                total_tokens_used = total_tokens_used + ?\n            WHERE user_id = ?\n        \"\"\", (datetime.now().isoformat(), tokens_used, user_id))
    else:
        cursor.execute(\"\"\"\n            UPDATE user_stats \n            SET total_generations = total_generations + 1,\n                failed_generations = failed_generations + 1,\n                last_generation_time = ?,\n                total_tokens_used = total_tokens_used + ?\n            WHERE user_id = ?\n        \"\"\", (datetime.now().isoformat(), tokens_used, user_id))
    
    conn.commit()
    conn.close()
    print(f"📊 Logged generation: {user_id} - Success: {success}")

"""
    
    content = content.replace(
        "story_states: Dict[str, Dict] = {}",
        "story_states: Dict[str, Dict] = {}" + db_functions
    )
    
    # Add stats endpoints before health check
    stats_endpoints = """
@app.get("/api/echo/stats/global")
async def get_global_stats():
    \"\"\"Get global statistics for admin overview\"\"\"
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total users
        cursor.execute("SELECT COUNT(*) FROM user_stats")
        total_users = cursor.fetchone()[0]
        
        # Total generations
        cursor.execute("SELECT SUM(total_generations) FROM user_stats")
        total_generations = cursor.fetchone()[0] or 0
        
        # Total successful generations
        cursor.execute("SELECT SUM(successful_generations) FROM user_stats")
        total_successful = cursor.fetchone()[0] or 0
        
        # Total failed generations
        cursor.execute("SELECT SUM(failed_generations) FROM user_stats")
        total_failed = cursor.fetchone()[0] or 0
        
        # Active users (last 24 hours)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM user_stats WHERE last_active > ?", (yesterday,))
        active_users_24h = cursor.fetchone()[0]
        
        # Most active users (top 5)
        cursor.execute(\"\"\"\n            SELECT user_id, total_generations, successful_generations, last_active\n            FROM user_stats \n            ORDER BY total_generations DESC \n            LIMIT 5\n        \"\"\")
        top_users = [dict(zip([desc[0] for desc in cursor.description], row)) 
                    for row in cursor.fetchall()]
        
        conn.close()
        
        # Calculate metrics
        success_rate = (total_successful / total_generations * 100) if total_generations > 0 else 0
        error_rate = (total_failed / total_generations * 100) if total_generations > 0 else 0
        
        return {
            "overview": {
                "total_users": total_users,
                "active_users_24h": active_users_24h,
                "total_generations": total_generations,
                "total_successful_generations": total_successful,
                "total_failed_generations": total_failed
            },
            "metrics": {
                "global_success_rate_percentage": round(success_rate, 2),
                "global_error_rate_percentage": round(error_rate, 2),
                "average_generations_per_user": round(total_generations / total_users, 2) if total_users > 0 else 0
            },
            "top_users": top_users,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving global stats: {str(e)}")

@app.get("/api/echo/stats/{user_id}")
async def get_user_stats(user_id: str = Path(..., description="User ID to get stats for")):
    \"\"\"Get comprehensive statistics for a specific user\"\"\"
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get basic user stats
        cursor.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
        user_row = cursor.fetchone()
        
        if not user_row:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Convert to dict
        columns = [desc[0] for desc in cursor.description]
        user_stats = dict(zip(columns, user_row))
        
        # Calculate success rate
        total_gens = user_stats[ total_generations ]
        success_rate = (user_stats[ successful_generations ] / total_gens * 100) if total_gens > 0 else 0
        error_rate = (user_stats[ failed_generations ] / total_gens * 100) if total_gens > 0 else 0
        
        # Get recent generation history (last 10)
        cursor.execute(\"\"\"\n            SELECT timestamp, prompt, success, image_path, processing_time\n            FROM generation_log \n            WHERE user_id = ? \n            ORDER BY timestamp DESC \n            LIMIT 10\n        \"\"\", (user_id,))
        recent_generations = [dict(zip([desc[0] for desc in cursor.description], row)) 
                            for row in cursor.fetchall()]
        
        # Get usage patterns (generations per day for last 7 days)
        seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute(\"\"\"\n            SELECT DATE(timestamp) as date, COUNT(*) as count\n            FROM generation_log \n            WHERE user_id = ? AND timestamp > ?\n            GROUP BY DATE(timestamp)\n            ORDER BY date\n        \"\"\", (user_id, seven_days_ago))
        daily_usage = [dict(zip([desc[0] for desc in cursor.description], row)) 
                      for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "user_id": user_id,
            "basic_stats": user_stats,
            "calculated_metrics": {
                "success_rate_percentage": round(success_rate, 2),
                "error_rate_percentage": round(error_rate, 2),
                "average_tokens_per_generation": round(user_stats[ total_tokens_used ] / total_gens, 2) if total_gens > 0 else 0
            },
            "recent_generations": recent_generations,
            "usage_patterns": {
                "daily_usage_last_7_days": daily_usage
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

"""
    
    content = content.replace(
        "@app.get(\"/api/echo/health\")",
        stats_endpoints + "@app.get(\"/api/echo/health\")"
    )
    
    # Update chat endpoint to include stats tracking
    content = content.replace(
        "    user_id = message.user_id\n    session_id = message.session_id\n    text = message.message",
        "    user_id = message.user_id\n    session_id = message.session_id\n    text = message.message\n    start_time = time.time()\n\n    # Ensure user stats exist and update activity\n    get_or_create_user_stats(user_id)\n    update_user_activity(user_id)"
    )
    
    # Update generation tracking
    content = content.replace(
        "            if image_path:\n                action_taken = \"IMAGE_GENERATED\"\n            else:\n                action_taken = \"IMAGE_GENERATION_FAILED\"",
        "            if image_path:\n                action_taken = \"IMAGE_GENERATED\"\n                processing_time = time.time() - start_time\n                log_generation_attempt(user_id, session_id, text, True, None, image_path, processing_time)\n            else:\n                action_taken = \"IMAGE_GENERATION_FAILED\"\n                processing_time = time.time() - start_time\n                log_generation_attempt(user_id, session_id, text, False, \"ComfyUI generation failed\", None, processing_time)"
    )
    
    # Update startup code
    content = content.replace(
        "    print(\"🚀 Starting Echo Brain Working Service on port 8309\")\n    print(\"🎨 ComfyUI integration enabled\")\n    print(\"📊 Story state tracking enabled\")",
        "    print(\"🚀 Starting Echo Brain Working Service with Stats on port 8309\")\n    print(\"🎨 ComfyUI integration enabled\")\n    print(\"📊 Story state tracking enabled\")\n    print(\"📈 User statistics tracking enabled\")\n    \n    # Initialize database\n    init_database()"
    )
    
    # Save the integrated file
    with open("echo_working_integrated.py", "w") as f:
        f.write(content)
    
    print("✅ Stats tracking integrated into echo_working_integrated.py")

if __name__ == "__main__":
    integrate_stats()

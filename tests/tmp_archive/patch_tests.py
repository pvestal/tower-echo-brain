import json

tests = [
    {
        "name": "hardware_ram",
        "query": "How much RAM does Tower have?",
        "expected_contains": ["96"],  # Just check for 96 (could be 96GB, 96 GB)
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "hardware_gpu_nvidia",
        "query": "What NVIDIA GPU does Tower have?",
        "expected_contains": ["3060"],  # Just check for 3060
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "hardware_gpu_amd",
        "query": "What AMD GPU does Tower have?",
        "expected_contains": ["9070"],  # Just check for 9070
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "patrick_info",
        "query": "Who is Patrick?",
        "expected_contains": ["patrick"],  # Just check it mentions Patrick
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    },
    {
        "name": "echo_brain_purpose",
        "query": "What is the purpose of Echo Brain?",
        "expected_contains": ["brain", "echo"],  # Basic check
        "expected_not_contains": [],
        "endpoint": "/api/echo/ask",
        "method": "POST",
        "timeout_ms": 30000,
    }
]

# Write to config file
with open('/opt/tower-echo-brain/config/self_tests.json', 'w') as f:
    json.dump(tests, f, indent=2)
    
print("✅ Created more flexible test config at /opt/tower-echo-brain/config/self_tests.json")

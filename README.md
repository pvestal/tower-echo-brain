# Echo Brain

## Structure
```
/opt/tower-echo-brain/
├── src/           # Source code
│   └── main.py    # Entry point (port 8309)
├── venv/          # Virtual environment
├── logs/          # Log files
└── archive/       # Everything else
```

## Run
```bash
systemctl status tower-echo-brain
```

## Test
```bash
curl -X POST http://localhost:8309/api/echo/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello"}'
```

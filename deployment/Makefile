.PHONY: test lint format run docker-build docker-run clean

test:
	pytest tests/ -v --cov=src

lint:
	black --check src/
	isort --check-only src/
	flake8 src/

format:
	black src/
	isort src/

run:
	uvicorn src.main_refactored:app --reload --host 0.0.0.0 --port 8309

docker-build:
	docker build -t tower-echo-brain:latest .

docker-run:
	docker run -p 8309:8309 tower-echo-brain:latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

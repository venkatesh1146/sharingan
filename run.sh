#!/bin/bash
# FastAPI Server Startup Script

# Load only host/port from .env to avoid JSON parsing issues in shell
API_HOST=$(grep -E '^API_HOST=' .env | cut -d= -f2-)
API_PORT=$(grep -E '^API_PORT=' .env | cut -d= -f2-)

# Activate virtual environment
source .venv/bin/activate

# Run the FastAPI server with uvicorn using env vars
uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000} --reload

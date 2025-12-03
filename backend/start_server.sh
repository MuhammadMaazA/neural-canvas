#!/bin/bash
# Start Neural Canvas Backend API

cd /cs/student/projects1/2023/muhamaaz/neural-canvas
source venv/bin/activate

cd backend

echo "Starting Neural Canvas API Server..."
echo "API will be available at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
echo ""

python main.py


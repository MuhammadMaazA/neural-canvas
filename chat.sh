#!/bin/bash
# Easy launcher for interactive chat with your model

cd /cs/student/projects1/2023/muhamaaz/neural-canvas
source venv/bin/activate
python llm/scripts/interactive_chat.py "$@"

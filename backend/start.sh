#!/bin/bash
export GROQ_API_KEY="***REMOVED***"
export ANTHROPIC_API_KEY="***REMOVED***"
export ELEVENLABS_API_KEY="***REMOVED***"
cd /var/www/qwen3-tts/backend
exec /var/www/qwen3-tts/backend/venv/bin/gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:4728 --timeout 900 app.main:app

#!/usr/bin/env bash
set -e

echo "Starting Smart Developer demo platform..."

echo ""
echo "1. Starting Postgres..."
docker compose up -d postgres

echo ""
echo "2. Start the algorithm service in a separate terminal:"
echo "   uvicorn algorithm.src.serving.api:app --host 0.0.0.0 --port 8001"

echo ""
echo "3. Start the backend gateway in another terminal:"
echo "   uvicorn backend.app.main:app --host 0.0.0.0 --port 8002"

echo ""
echo "4. Start the frontend:"
echo "   cd frontend && npm run dev"

echo ""
echo "Then open:"
echo "   http://localhost:5173"
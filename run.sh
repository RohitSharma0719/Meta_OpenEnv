set -e
cd "$(dirname "$0")"

echo "Starting Support Triage Environment..."

# Kill anything already using port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
sleep 1

uv run server --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

echo "Waiting for server to start..."
for i in {1..20}; do
    if curl -s -X POST http://localhost:8000/reset > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 1
done

echo ""
echo "Running inference agent..."
echo "================================"
uv run python inference.py

echo ""
echo "Shutting down server..."
kill $SERVER_PID 2>/dev/null
echo "Done!"

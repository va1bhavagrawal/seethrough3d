#!/bin/bash

# Accept ports from command line or use defaults
PORT_CV=${1:-5001}
PORT_FINAL=${2:-5002}
PORT_SEGMASK=${3:-5003}
PORT_PAPER=${4:-5004}

PORTS=("$PORT_CV" "$PORT_FINAL" "$PORT_SEGMASK" "$PORT_PAPER")

for port in "${PORTS[@]}"; do
  PID=$(lsof -t -i ":$port")
  if [ -n "$PID" ]; then
    echo "Killing process $PID running on port $port..."
    kill -9 "$PID"
    echo "Process $PID killed."
  else
    echo "No process found running on port $port."
  fi
done

# Start CV render server
echo "Starting Camera View render server on port $PORT_CV..."
python blender_server.py --mode cv --port "$PORT_CV" &
CV_PID=$!

echo "Starting Final (Cycles) render server on port $PORT_FINAL..."
python blender_server.py --mode final --port "$PORT_FINAL" &
FINAL_PID=$!

# Start segmask render server
echo "Starting Segmentation Mask render server on port $PORT_SEGMASK..."
python3 blender_server_segmasks.py --port "$PORT_SEGMASK" &
SEGMASK_PID=$!

echo "Starting Paper render server on port $PORT_PAPER..."
python blender_server.py --mode paper --port "$PORT_PAPER" &
PAPER_PID=$!

echo "Render servers started!"
echo "CV Server PID: $CV_PID (port $PORT_CV)"
echo "Final (Cycles) Render Server PID: $FINAL_PID (port $PORT_FINAL)"
echo "Segmentation Mask Server PID: $SEGMASK_PID (port $PORT_SEGMASK)"
echo "Paper Render Server PID: $PAPER_PID (port $PORT_PAPER)"

# Function to cleanup on exit
cleanup() {
    echo "Stopping render servers..."
    kill $CV_PID $FINAL_PID $SEGMASK_PID $PAPER_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait $CV_PID $FINAL_PID $SEGMASK_PID  
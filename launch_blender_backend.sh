#!/bin/bash

PORTS=(5001 5002 5003 5004)

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
echo "Starting Camera View render server on port 5001..."
python blender_server.py --mode cv --port 5001 &
CV_PID=$!

echo "Starting Camera View render server on port 5002..."
python blender_server.py --mode final --port 5002 &
FINAL_PID=$!


# Start segmask render server
echo "Starting Segmentation Mask render server on port 5003..."
python3 blender_server_segmasks.py --port 5003 &
SEGMASK_PID=$!

echo "Starting Camera View render server on port 5004..."
python blender_server.py --mode paper --port 5004 &
PAPER_PID=$!

echo "Render servers started!"
echo "CV Server PID: $CV_PID (port 5001)"
echo "Final (Cycles) Render Server PID: $FINAL_PID (port 5002)"
echo "Segmentation Mask Server PID: $SEGMASK_PID (port 5003)"

# Function to cleanup on exit
cleanup() {
    echo "Stopping render servers..."
    kill $CV_PID $FINAL_PID $SEGMASK_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait $CV_PID $FINAL_PID $SEGMASK_PID  
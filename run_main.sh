#!/bin/bash
PORT=8602
PID=$(lsof -ti :$PORT) 

if [ ! -z "$PID" ]; then
  echo "Killing process on port $PORT..."
  kill -9 $PID
fi

echo "Starting Streamlit on port $PORT..."
streamlit run main.py --server.port $PORT
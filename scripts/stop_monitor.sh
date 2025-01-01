#!/bin/bash

if [ -f ".monitor.pid" ]; then
    PID=$(cat .monitor.pid)
    if ps -p $PID > /dev/null; then
        echo "Stopping Obsidian monitor (PID: $PID)..."
        kill $PID
        rm .monitor.pid
        echo "Monitor stopped"
    else
        echo "Monitor process not found"
        rm .monitor.pid
    fi
else
    echo "Monitor PID file not found"
fi 
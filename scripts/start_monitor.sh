#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the Obsidian monitor in the background
echo "Starting Obsidian monitor..."
nohup python utils/obsidian_monitor.py > logs/obsidian_monitor.log 2>&1 &

# Save the PID to a file for later cleanup
echo $! > .monitor.pid

echo "Obsidian monitor started (PID: $!)"
echo "Monitor logs available at: logs/obsidian_monitor.log" 
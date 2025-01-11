import os
import time
import psutil
from pathlib import Path
import subprocess
import signal
import sys
from datetime import datetime
from dotenv import load_dotenv
import logging
import google.cloud.logging
from google.cloud.logging_v2.handlers import CloudLoggingHandler

# Load environment variables
load_dotenv()

# Initialize GCP logger
def setup_logging_for_gcp(logger_name: str = "mirror-agent") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    # GCP
    client = google.cloud.logging.Client()
    cloud_handler = CloudLoggingHandler(client, name=logger_name)
    logger.addHandler(cloud_handler)

    # Also console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging_for_gcp("mirror-agent")

# Check if running in VS Code
IN_VSCODE = os.environ.get('VSCODE_CLI', False) or os.environ.get('TERM_PROGRAM') == 'vscode'

def is_obsidian_running():
    """Check if Obsidian is running by looking for the process."""
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] == 'Obsidian':
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def run_indexer():
    """Run the obsidian_indexer.py script."""
    try:
        script_path = Path(__file__).parent / 'obsidian_indexer.py'
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent.parent)
        
        logger.info("Starting Obsidian indexer", {
            "trigger": "obsidian_app_opened",
            "timestamp": datetime.now().isoformat()
        })
        
        # Print status for VS Code
        if IN_VSCODE:
            print("\n[Obsidian Monitor] Running indexer...")
        
        proc = subprocess.Popen(
            ['python', str(script_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        for line in proc.stdout:
            # Stream each line as INFO
            logger.info("[indexer] %s", line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            logger.error("obsidian_indexer.py exited with status %d", proc.returncode)
        else:
            logger.info("Indexer completed successfully", {
                "timestamp": datetime.now().isoformat()
            })
            if IN_VSCODE:
                print("[Obsidian Monitor] Indexing completed successfully")
            
    except Exception as e:
        logger.error(e, {"context": "running_indexer"})
        if IN_VSCODE:
            print(f"[Obsidian Monitor] Error running indexer: {str(e)}")

def monitor_obsidian():
    """Monitor for Obsidian app activity and trigger indexer when opened."""
    if IN_VSCODE:
        print("\n[Obsidian Monitor] Starting monitor in VS Code...")
    else:
        print("Starting Obsidian monitor...")
        
    logger.info("Starting Obsidian monitor", {
        "timestamp": datetime.now().isoformat(),
        "context": "vscode" if IN_VSCODE else "shell"
    })
    
    was_running = is_obsidian_running()
    
    def handle_shutdown(signum, frame):
        if IN_VSCODE:
            print("\n[Obsidian Monitor] Shutting down...")
        else:
            print("\nShutting down Obsidian monitor...")
            
        logger.info("Shutting down Obsidian monitor", {
            "timestamp": datetime.now().isoformat(),
            "context": "vscode" if IN_VSCODE else "shell"
        })
        sys.exit(0)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    try:
        while True:
            is_running = is_obsidian_running()
            
            # If Obsidian wasn't running before but is now, it was just opened
            if not was_running and is_running:
                if IN_VSCODE:
                    print("\n[Obsidian Monitor] Obsidian opened, triggering indexer...")
                else:
                    print("\nObsidian opened, triggering indexer...")
                run_indexer()
            
            was_running = is_running
            time.sleep(5)  # Check every 5 seconds
            
    except Exception as e:
        logger.error(e, {"context": "obsidian_monitor"})
        if IN_VSCODE:
            print(f"[Obsidian Monitor] Error: {str(e)}")
        else:
            print(f"Error in monitor: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    monitor_obsidian() 
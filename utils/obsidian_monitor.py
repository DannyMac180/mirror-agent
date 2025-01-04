import os
import time
import psutil
from pathlib import Path
import subprocess
import signal
import sys
from datetime import datetime
from dotenv import load_dotenv
from . import gcp_logging

# Load environment variables
load_dotenv()

# Initialize GCP logger
logger = gcp_logging.get_logger("mirror-agent")

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
        
        logger.log_info("Starting Obsidian indexer", {
            "trigger": "obsidian_app_opened",
            "timestamp": datetime.now().isoformat()
        })
        
        # Print status for VS Code
        if IN_VSCODE:
            print("\n[Obsidian Monitor] Running indexer...")
        
        result = subprocess.run(
            ['python', str(script_path)],
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.log_info("Indexer completed successfully", {
                "stdout": result.stdout[-500:],  # Last 500 chars
                "timestamp": datetime.now().isoformat()
            })
            if IN_VSCODE:
                print("[Obsidian Monitor] Indexing completed successfully")
        else:
            logger.log_error(Exception(result.stderr), {
                "context": "indexer_execution",
                "stdout": result.stdout[-500:],
                "stderr": result.stderr[-500:]
            })
            if IN_VSCODE:
                print(f"[Obsidian Monitor] Error during indexing: {result.stderr[-200:]}")
            
    except Exception as e:
        logger.log_error(e, {"context": "running_indexer"})
        if IN_VSCODE:
            print(f"[Obsidian Monitor] Error running indexer: {str(e)}")

def monitor_obsidian():
    """Monitor for Obsidian app activity and trigger indexer when opened."""
    if IN_VSCODE:
        print("\n[Obsidian Monitor] Starting monitor in VS Code...")
    else:
        print("Starting Obsidian monitor...")
        
    logger.log_info("Starting Obsidian monitor", {
        "timestamp": datetime.now().isoformat(),
        "context": "vscode" if IN_VSCODE else "shell"
    })
    
    was_running = is_obsidian_running()
    
    def handle_shutdown(signum, frame):
        if IN_VSCODE:
            print("\n[Obsidian Monitor] Shutting down...")
        else:
            print("\nShutting down Obsidian monitor...")
            
        logger.log_info("Shutting down Obsidian monitor", {
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
        logger.log_error(e, {"context": "obsidian_monitor"})
        if IN_VSCODE:
            print(f"[Obsidian Monitor] Error: {str(e)}")
        else:
            print(f"Error in monitor: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    monitor_obsidian() 
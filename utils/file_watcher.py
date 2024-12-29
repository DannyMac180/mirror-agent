from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from pathlib import Path
from create_code_snapshot import create_code_snapshot, get_gitignore_spec

class SnapshotHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_snapshot_time = 0
        self.cooldown = 2  # Cooldown in seconds to prevent multiple rapid snapshots
        self.gitignore_spec = get_gitignore_spec()
        self.project_root = Path(__file__).parent.parent

    def on_modified(self, event):
        # Skip directory modifications and non-file events
        if event.is_directory:
            return
            
        # Convert path to relative for gitignore matching
        try:
            file_path = Path(event.src_path)
            rel_path = file_path.relative_to(self.project_root)
            
            # Skip gitignored files
            if self.gitignore_spec and self.gitignore_spec.match_file(str(rel_path)):
                return
                
            current_time = time.time()
            # Check if we're still in cooldown period
            if current_time - self.last_snapshot_time < self.cooldown:
                return
                
            print(f"\nFile change detected: {rel_path}")
            create_code_snapshot()  # Snapshot entire project
            self.last_snapshot_time = current_time
        except ValueError:
            # Path is not relative to project root, ignore it
            return

def start_watcher():
    """Start watching the project directory for changes and create snapshots."""
    event_handler = SnapshotHandler()
    observer = Observer()
    
    # Watch the entire project directory
    project_root = Path(__file__).parent.parent
    print(f"Starting to watch project root: {project_root}")
    observer.schedule(event_handler, str(project_root), recursive=True)

    observer.start()
    print("\nWatcher started. Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nWatcher stopped.")
    
    observer.join()

if __name__ == "__main__":
    start_watcher() 
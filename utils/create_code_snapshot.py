import os
from pathlib import Path
import pathspec

def get_gitignore_spec() -> pathspec.PathSpec:
    """Load .gitignore patterns and create a PathSpec matcher."""
    # Get project root (parent of utils directory)
    project_root = Path(__file__).parent.parent
    gitignore_path = project_root / ".gitignore"
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            return pathspec.PathSpec.from_lines('gitwildmatch', f)
    return pathspec.PathSpec([])

def create_code_snapshot(output_file: str = "full_code_snapshot.txt") -> None:
    """Create a snapshot of all code files in the project, respecting .gitignore."""
    
    # Get project root (parent of utils directory)
    project_root = Path(__file__).parent.parent
    
    # Get gitignore patterns
    gitignore = get_gitignore_spec()
    
    # Important directories to include
    important_dirs = ['src', 'tests', 'docs', 'utils', 'tools']
    
    # Track total files processed
    total_files = 0
    
    # Open output file
    with open(output_file, 'w', encoding='utf-8') as out:
        # First process important directories
        for dir_name in important_dirs:
            dir_path = project_root / dir_name
            if not dir_path.exists():
                continue
                
            for root, dirs, files in os.walk(dir_path):
                # Skip the entire directory if it matches gitignore
                rel_root = Path(root).relative_to(project_root)
                if gitignore.match_file(str(rel_root)):
                    dirs.clear()  # Don't descend into ignored directories
                    continue
                    
                for file in sorted(files):  # Sort files for consistent output
                    file_path = Path(root) / file
                    
                    # Get path relative to project root for gitignore matching
                    rel_path = file_path.relative_to(project_root)
                    
                    # Skip if matches gitignore patterns
                    if gitignore.match_file(str(rel_path)):
                        continue
                        
                    # Only include code files
                    if file.endswith(('.py', '.md', '.txt', '.yml', '.yaml', '.json', '.toml')):
                        try:
                            # Write file header
                            out.write(f"\n{'='*80}\n")
                            out.write(f"File: {rel_path}\n")
                            out.write(f"{'='*80}\n\n")
                            
                            # Write file contents
                            with open(file_path, 'r', encoding='utf-8') as f:
                                out.write(f.read())
                            
                            total_files += 1
                            print(f"Added: {rel_path}")
                                
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")
        
        # Then process root directory files
        for file in sorted(os.listdir(project_root)):
            file_path = project_root / file
            
            # Skip directories and already processed files
            if file_path.is_dir() or file in [output_file]:
                continue
                
            # Get path relative to project root for gitignore matching
            rel_path = file_path.relative_to(project_root)
            
            # Skip if matches gitignore patterns
            if gitignore.match_file(str(rel_path)):
                continue
                
            # Only include code files
            if file.endswith(('.py', '.md', '.txt', '.yml', '.yaml', '.json', '.toml')):
                try:
                    # Write file header
                    out.write(f"\n{'='*80}\n")
                    out.write(f"File: {rel_path}\n")
                    out.write(f"{'='*80}\n\n")
                    
                    # Write file contents
                    with open(file_path, 'r', encoding='utf-8') as f:
                        out.write(f.read())
                    
                    total_files += 1
                    print(f"Added: {rel_path}")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

    print(f"\nProcessed {total_files} files successfully!")

if __name__ == "__main__":
    create_code_snapshot()
    print("Code snapshot created successfully!")
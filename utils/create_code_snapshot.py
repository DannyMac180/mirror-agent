import os
from pathlib import Path
import pathspec

def create_code_snapshot(output_file: str = "full_code_snapshot.txt") -> None:
    """Create a snapshot of all code files in the project, respecting .gitignore."""
    
    # Read .gitignore patterns
    gitignore_path = Path(__file__).parent.parent / ".gitignore"
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            gitignore = pathspec.PathSpec.from_lines('gitwildmatch', f)
    else:
        gitignore = pathspec.PathSpec([])
    
    # Get project root (parent of utils directory)
    project_root = Path(__file__).parent.parent
    
    # Open output file
    with open(output_file, 'w', encoding='utf-8') as out:
        # Walk through all files in project
        for root, _, files in os.walk(project_root):
            for file in files:
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
                            
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    create_code_snapshot()
    print("Code snapshot created successfully!")
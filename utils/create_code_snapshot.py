import os
from pathlib import Path
import pathspec

def get_gitignore_spec():
    """Load .gitignore patterns and create a PathSpec matcher."""
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        return None
        
    with open(gitignore_path, "r") as f:
        gitignore = f.read().splitlines()
    
    return pathspec.PathSpec.from_lines('gitwildmatch', gitignore)

def create_code_snapshot(
    directories=None,
    output_file="full_code_snapshot.txt",
):
    """
    Recursively collects all non-gitignored files from the specified directories and concatenates them.
    If no directories specified, scans the entire project root (except gitignored paths).

    Args:
        directories (list, optional): A list of directories to scan recursively. If None, scans project root.
        output_file (str): The path to the output file containing the concatenated code.
        
    Returns:
        str: The path to the generated code snapshot file.
    """
    # Get gitignore rules
    gitignore_spec = get_gitignore_spec()
    
    # If no directories specified, use project root
    if directories is None:
        root_dir = Path().resolve()
        directories = [root_dir]
    else:
        directories = [Path(d).resolve() for d in directories]

    # Prepare output file
    output_path = Path(output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for directory in directories:
            if not directory.exists():
                print(f"Warning: Directory {directory} does not exist, skipping.")
                continue
            
            # Walk through the directory structure
            for root, dirs, files in os.walk(directory):
                root_path = Path(root)
                
                # Remove ignored directories from the walk
                if gitignore_spec:
                    dirs[:] = [d for d in dirs if not gitignore_spec.match_file(str(root_path / d))]
                
                # Sort files for consistency
                files.sort()

                for file_name in files:
                    file_path = root_path / file_name
                    rel_path = file_path.relative_to(directory.parent)
                    
                    # Skip files that match gitignore patterns
                    if gitignore_spec and gitignore_spec.match_file(str(rel_path)):
                        continue
                        
                    # Skip the snapshot file itself
                    if file_path == output_path:
                        continue

                    # Write a header to indicate the start of this file's content
                    out_f.write(f"\n\n# ======= File: {rel_path} =======\n\n")

                    # Read and append file content
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        out_f.write(content)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
                        continue
    
    print(f"Code snapshot created at: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    # Create snapshot of entire project (except gitignored files)
    snapshot_path = create_code_snapshot()
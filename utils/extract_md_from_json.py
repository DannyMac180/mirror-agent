import json
import os
from pathlib import Path

def extract_markdown_from_json_files(directory: str) -> str:
    """Extract markdown content from all JSON files in the directory."""
    all_markdown = []
    
    # Get all JSON files in the directory
    json_files = sorted([f for f in os.listdir(directory) if f.endswith('.json')])
    
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        print(f"Processing {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract markdown content if it exists
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'markdown' in item:
                            all_markdown.append(item['markdown'])
                elif isinstance(data, dict) and 'markdown' in data:
                    all_markdown.append(data['markdown'])
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    return "\n\n".join(all_markdown)

def main():
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs" / "langgraph"
    
    # Extract all markdown content
    combined_markdown = extract_markdown_from_json_files(str(docs_dir))
    
    # Save to a new markdown file
    output_file = docs_dir / "combined_documentation.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(combined_markdown)
    
    print(f"Combined markdown has been saved to {output_file}")

if __name__ == "__main__":
    main()

"""Tool for generating documentation by crawling documentation websites.

This tool can either crawl from a provided URL or search for official documentation.
Uses Firecrawl for deep crawling and falls back to direct fetching if no API key is available.
"""

import os
import sys
from typing import Optional, List
from urllib.parse import urlparse
from pathlib import Path

from firecrawl import FirecrawlApp
import requests
from bs4 import BeautifulSoup, NavigableString
from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader

# Load environment variables
load_dotenv()

# Initialize Firecrawl client if possible
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
app = FirecrawlApp(api_key=FIRECRAWL_API_KEY) if FIRECRAWL_API_KEY else None

def find_docs_url(topic: str) -> str:
    """Search for the official documentation URL for a given topic.
    
    Args:
        topic: The framework or topic to find docs for
        
    Returns:
        str: The URL of the official documentation
    """
    search_query = f"{topic} official documentation"
    response = requests.get(
        "https://www.google.com/search",
        params={"q": search_query},
        headers={"User-Agent": "Mozilla/5.0"}
    )
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find first result that looks like documentation
    for result in soup.find_all("a"):
        href = result.get("href", "")
        if "url?q=" in href and not "google" in href:
            url = href.split("url?q=")[1].split("&")[0]
            # Check if URL seems like documentation
            parsed = urlparse(url)
            if any(x in parsed.netloc for x in ["docs.", ".dev", ".io", "github"]):
                return url
                
    raise ValueError(f"Could not find documentation URL for {topic}")

def crawl_with_firecrawl(url: str) -> List[str]:
    """Crawl documentation using Firecrawl.
    
    Args:
        url: The documentation URL to crawl
        
    Returns:
        List[str]: List of markdown content from crawled pages
    """
    crawl_status = app.crawl_url(
        url,
        params={
            "limit": 1000,  # Page limit as specified
            "scrapeOptions": {"formats": ["markdown", "html"]}
        },
        poll_interval=30
    )
    
    markdown_content = []
    for page in crawl_status.pages:
        if page.markdown:
            markdown_content.append(page.markdown)
            
    return markdown_content

def fetch_direct(url: str) -> str:
    """Fetch documentation content directly from a URL.
    
    Args:
        url: The documentation URL to fetch
        
    Returns:
        str: The markdown content
    """
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove unwanted elements
    for element in soup.find_all(class_=["feedback", "nav", "footer", "sidebar", "pagination", "breadcrumb", "toc"]):
        element.decompose()
    
    # Remove script, style tags and feedback elements
    for element in soup.find_all(['script', 'style']):
        element.decompose()
    
    # Remove feedback section and other unwanted elements by text content
    for element in soup.find_all(lambda tag: tag.string and any(x in str(tag.string).lower() for x in ['feedback', 'was this page helpful', 'last modified', 'yes', 'no'])):
        element.decompose()
        
    # Extract main content
    content = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
    if not content:
        return response.text
    
    # Process content in order of appearance
    formatted_content = []
    seen_content = set()
    
    def clean_text(text):
        """Clean and normalize text content."""
        # Remove extra whitespace and normalize spaces
        text = ' '.join(text.split())
        # Remove common duplicated content markers
        text = text.replace('Documentation Documentation', 'Documentation')
        return text.strip()
    
    def add_content(text, prefix=""):
        """Helper to add unique content with optional prefix."""
        text = clean_text(text)
        if text and text not in seen_content and len(text) > 1:
            seen_content.add(text)
            if prefix:
                formatted_content.append(f"\n{prefix}{text}\n")
            else:
                formatted_content.append(text + "\n")
    
    # First pass: Extract main description
    main_desc = content.find('p')
    if main_desc:
        add_content(main_desc.get_text())
    
    # Second pass: Process headers and content in order
    for element in content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'ul', 'ol']):
        # Skip empty or very short elements
        if not element.get_text(strip=True) or len(element.get_text(strip=True)) < 2:
            continue
            
        # Handle headers
        if element.name.startswith('h'):
            level = int(element.name[1])
            text = element.get_text(strip=True)
            if text:
                add_content(text, '#' * level + ' ')
        # Handle paragraphs and other content
        elif element.name in ['p', 'div']:
            text = element.get_text(separator=' ', strip=True)
            if text and not any(text.lower() in seen.lower() for seen in seen_content):
                add_content(text)
        # Handle lists
        elif element.name in ['ul', 'ol']:
            for item in element.find_all('li'):
                text = item.get_text(strip=True)
                if text:
                    add_content(f"- {text}")
    
    return "\n".join(formatted_content)

def generate_docs(topic_or_url: str, output_file: Optional[str] = None) -> str:
    """Generate documentation by crawling from a URL or searching for docs.
    
    Args:
        topic_or_url: Either a direct URL to crawl from or a topic to search for
        output_file: Optional custom output path
        
    Returns:
        str: Path to the generated markdown file
    """
    # Determine if input is URL or topic
    is_url = any(topic_or_url.startswith(prefix) for prefix in ["http://", "https://"])
    
    # Get docs URL
    url = topic_or_url if is_url else find_docs_url(topic_or_url)
        
    # Generate output path
    if output_file is None:
        if is_url:
            parsed = urlparse(url)
            clean_name = parsed.netloc.replace(".", "_") + parsed.path.replace("/", "_")
        else:
            clean_name = topic_or_url.lower().replace(" ", "_")
        output_file = f"docs/{clean_name}_docs.md"
    
    # Try Firecrawl first, fall back to direct fetch
    if app:
        try:
            content = crawl_with_firecrawl(url)
            content = "\n\n".join(content)
        except Exception as e:
            print(f"Firecrawl failed: {e}. Falling back to direct fetch...")
            content = fetch_direct(url)
    else:
        print("No Firecrawl API key found. Using direct fetch...")
        content = fetch_direct(url)
    
    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        title = topic_or_url if is_url else topic_or_url.title()
        f.write(f"# {title} Documentation\n\n")
        f.write(f"Source: {url}\n\n")
        f.write(content)
    
    return output_file

def main():
    if len(sys.argv) != 2:
        print("Usage: python docs_generator.py <url>")
        sys.exit(1)
        
    url = sys.argv[1]
    try:
        # Initialize FireCrawl loader in crawl mode
        loader = FireCrawlLoader(
            api_key=os.getenv("FIRECRAWL_API_KEY"),
            url=url,
            mode="crawl"  # Use crawl mode to get all subpages
        )
        
        # Load all pages
        print(f"Crawling {url} and its subpages...")
        pages = loader.load()
        
        # Combine all page content
        combined_content = []
        for page in pages:
            combined_content.append(page.page_content)
            
        full_content = "\n\n".join(combined_content)
        
        # Create markdown file
        output_filename = url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_") + "_docs.md"
        output_path = Path("docs") / output_filename
        
        with open(output_path, "w") as f:
            f.write(f"# {url} Documentation\n\n")
            f.write(f"Source: {url}\n\n")
            f.write(full_content)
            
        print(f"Generated docs at: {output_path}")
        
    except Exception as e:
        print(f"Firecrawl failed: {str(e)}. Falling back to direct fetch...")
        # Fallback to direct fetch
        content = fetch_direct(url)
        
        # Create markdown file
        output_filename = url.replace("https://", "").replace("http://", "").replace("/", "_").replace(".", "_") + "_docs.md"
        output_path = Path("docs") / output_filename
        
        with open(output_path, "w") as f:
            f.write(f"# {url} Documentation\n\n")
            f.write(f"Source: {url}\n\n")
            f.write(content)
            
        print(f"Generated docs at: {output_path}")

if __name__ == "__main__":
    main()

"""Tool for generating documentation by crawling documentation websites.

This tool can either crawl from a provided URL or search for official documentation.
Uses Firecrawl for deep crawling and falls back to direct fetching if no API key is available.
"""

import os
from typing import Optional, List
from urllib.parse import urlparse

from firecrawl import FirecrawlApp
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

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
    
    # Extract main content
    content = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
    if not content:
        return response.text
        
    return content.get_text()

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

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python docs_generator.py <topic_or_url>")
        print("Examples:")
        print("  python docs_generator.py 'langchain pinecone integration'")
        print("  python docs_generator.py https://docs.example.com/api")
        sys.exit(1)
        
    topic_or_url = sys.argv[1]
    output_file = generate_docs(topic_or_url)
    print(f"Generated docs at: {output_file}")
    if not app:
        print("\nNote: Install Firecrawl API key in .env for better results:")
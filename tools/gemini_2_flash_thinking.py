#!/usr/bin/env python3

import argparse
from google import genai
import os
import sys

def stream_response(response):
    """Stream both thoughts and responses from the model."""
    full_response = ""
    thought_tokens = []

    print("\nStreaming response:\n")
    for chunk in response:
        for part in chunk.candidates[0].content.parts:
            if part.thought:
                thought = f"Model Thought: {part.text}"
                sys.stdout.write(f"\033[33m{thought}\033[0m\n")  # Yellow color for thought tokens
                sys.stdout.flush()
                thought_tokens.append(part.text)
            else:
                sys.stdout.write(part.text)
                sys.stdout.flush()
                full_response += part.text
    
    return full_response.strip(), thought_tokens

def get_gemini_response(prompt: str, model_name: str = "gemini-2.0-flash-thinking-exp-01-21") -> dict:
    # Configure the client with v1alpha API version
    client = genai.Client(
        api_key=os.getenv("GOOGLE_API_KEY"),
        http_options={'api_version': 'v1alpha'}
    )
    
    # Configure thinking settings
    config = {
        'thinking_config': {'include_thoughts': True},
        'temperature': 0.7,
        'safety_settings': [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }
    
    # Generate content with streaming
    response = client.models.generate_content_stream(
        model=model_name,
        contents=prompt,
        config=config
    )
    
    # Stream and collect response
    full_response, thought_tokens = stream_response(response)
            
    return {
        'response': full_response,
        'thinking_tokens': thought_tokens
    }

def main():
    parser = argparse.ArgumentParser(description='Query Gemini 2 Flash Thinking model')
    parser.add_argument('prompt', type=str, help='The prompt to send to the model')
    parser.add_argument('--model', type=str, default="gemini-2.0-flash-thinking-exp-01-21",
                      help='Model name (default: gemini-2.0-flash-thinking-exp-01-21)')
    
    args = parser.parse_args()
    
    try:
        result = get_gemini_response(args.prompt, args.model)
        
        print("\n\nFinal Response:")
        print(result['response'])
        
        if result['thinking_tokens']:
            print("\nCollected Thinking Tokens:")
            for token in result['thinking_tokens']:
                print(f"- {token}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
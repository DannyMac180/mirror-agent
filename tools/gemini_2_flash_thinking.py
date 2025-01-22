#!/usr/bin/env python3

import argparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks import get_openai_callback
import os
import json

def get_gemini_response(prompt, model_name="gemini-2.0-flash-thinking-exp-01-21"):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        convert_system_message_to_human=True,
        verbose=True
    )
    
    response = llm.invoke(prompt)
    
    # Extract thinking tokens from the response metadata
    thinking_tokens = response.additional_kwargs.get('thinking_tokens', [])
    
    return {
        'response': response.content,
        'thinking_tokens': thinking_tokens
    }

def main():
    parser = argparse.ArgumentParser(description='Query Gemini 2 Flash Thinking model')
    parser.add_argument('prompt', type=str, help='The prompt to send to the model')
    parser.add_argument('--model', type=str, default="gemini-2.0-flash-thinking-exp-01-21",
                      help='Model name (default: gemini-2.0-flash-thinking-exp-01-21)')
    
    args = parser.parse_args()
    
    try:
        result = get_gemini_response(args.prompt, args.model)
        
        print("\nResponse:")
        print(result['response'])
        print("\nThinking Tokens:")
        for token in result['thinking_tokens']:
            print(f"- {token}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
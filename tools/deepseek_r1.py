#!/usr/bin/env python3

import os
import sys
import argparse
from openai import OpenAI

def get_deepseek_response(prompt: str) -> tuple[str, str]:
    """Get response from DeepSeek reasoner model."""
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages
    )

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    return reasoning_content, content

def main():
    parser = argparse.ArgumentParser(description='Get response from DeepSeek reasoner')
    parser.add_argument('prompt', type=str, help='Prompt for the model')
    args = parser.parse_args()

    try:
        reasoning, response = get_deepseek_response(args.prompt)
        print("\nReasoning:")
        print("-" * 50)
        print(reasoning)
        print("\nResponse:")
        print("-" * 50)
        print(response)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

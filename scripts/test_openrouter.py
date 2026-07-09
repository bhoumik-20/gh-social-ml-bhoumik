#!/usr/bin/env python3
import os
import sys
import time
import requests

# Ensure project root is in the path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from utils.readme_processor import process_markdown

def main():
    print("🚀 OpenRouter API Inference Test Script")
    print("=======================================")

    # 1. Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in environment.")
        print("Please add 'OPENROUTER_API_KEY=your_key_here' to your .env file.")
        return

    # 2. Use a dummy text instead of querying the DB since DB is empty
    print("Using a test string as the source README text...")
    raw_text = """
    # Dummy Repo
    
    This is a test repository.
    
    ## Installation
    `pip install dummy`
    
    ## Usage
    Run the dummy script to do dummy things.
    """

    # Process and clean text
    clean_text = process_markdown(raw_text).clean_text
    print(f"Source text size: {len(clean_text)} characters.")

    # 3. Formulate OpenRouter prompt and payload
    model = os.getenv("OPENROUTER_MODEL_ID", "meta-llama/llama-3.3-70b-instruct")
    url = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
    
    prompt = (
        "You are an expert technical writer. Convert the following plain text version of a GitHub repository README "
        "into a clean, well-structured, and highly readable Markdown document. Use logical headings, subheadings, "
        "bullet points, and an improved overall structure.\n\n"
        "Instructions:\n"
        "- Do NOT add or introduce new facts, library features, or hallucinate information not present in the source text.\n"
        "- Preserve all important technical details, configurations, installation instructions, code snippets, and commands.\n"
        "- Remove unnecessary boilerplate or redundancies to make it clean and readable.\n"
        "- Return ONLY the generated Markdown text. Do NOT include any introductory or concluding comments, greetings, or conversational remarks.\n\n"
        f"Source text:\n{clean_text[:3000]}"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("CLIENT_URL") or "http://localhost:8081",
        "X-Title": "GH Social ML Test",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1
    }

    print(f"\nSending inference request to OpenRouter (Model: {model})...")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            res_json = response.json()
            content = res_json["choices"][0]["message"]["content"].strip()
            
            # Print performance metrics
            print("\n⚡ Performance Statistics:")
            print(f"  - Request Duration: {duration:.2f} seconds")
            if "usage" in res_json:
                usage = res_json["usage"]
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                print(f"  - Completion Tokens: {completion_tokens}")
                print(f"  - Prompt Tokens: {prompt_tokens}")
                if duration > 0 and completion_tokens > 0:
                    print(f"  - Throughput: {completion_tokens / duration:.1f} tokens/second")

            print("\n📝 Generated README Markdown:")
            print("-------------------------------------")
            print(content)
            print("-------------------------------------")
            
        else:
            print(f"❌ Error: OpenRouter API returned status {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    main()

import os
import sys
import time

# Resolve project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv()

from database import PostgreSQLConnector
from utils.gemma_client import generate_readme_markdown
from utils.readme_processor import process_markdown

def main():
    print("🚀 Running Gemma README Markdown Generation Test...")
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMMA_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY is not set in your .env file!")
        print("Please add it to your .env: GEMINI_API_KEY=your_key_here")
        return
        
    print(f"API Key detected. Using model: {os.getenv('GEMMA_MODEL_ID', 'gemini-2.5-flash')}")
    
    # Retrieve langflow description/summary from Supabase
    db = PostgreSQLConnector()
    if not db.enabled or not db.verify_connection():
        print("❌ Error: Could not connect to database.")
        return

    print("Fetching 'langflow-ai/langflow' description from Supabase...")
    conn = db.connect()
    cursor = conn.cursor()
    cursor.execute("SELECT description, readme_summary FROM Repo WHERE full_name = 'langflow-ai/langflow';")
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        print("❌ Error: langflow repo not found in database. Using default fallback description.")
        raw_text = "Langflow is a powerful, low-code interface for building RAG applications and AI agents."
    else:
        description, readme_summary = row
        raw_text = readme_summary or description

    # Process and clean text
    clean_text = process_markdown(raw_text).clean_text
    print(f"Source text size: {len(clean_text)} characters.")
    
    print("\nCalling Gemma model via Gemini Cloud API (this may take a few seconds)...")
    start_time = time.time()
    markdown_out = generate_readme_markdown(clean_text[:3000])
    duration = time.time() - start_time
    
    if markdown_out:
        print("\n✅ SUCCESS: Gemma generated the following Markdown:")
        print(f"  - Request Duration: {duration:.2f} seconds")
        print("------------------------")
        print(markdown_out)
        print("------------------------")
    else:
        print("\n❌ FAILURE: Gemma API did not return any markdown output. Check logs above for details.")

if __name__ == "__main__":
    main()


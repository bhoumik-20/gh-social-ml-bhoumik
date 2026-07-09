import logging
import os
import re
import time
import threading
import requests

logger = logging.getLogger(__name__)

class OpenRouterRateLimiter:
    """Thread-safe rate limiter to prevent exceeding the OpenRouter API requests-per-minute (RPM) quota."""
    def __init__(self, rpm_limit: float = 100.0):
        self.rpm_limit = rpm_limit
        self.lock = threading.Lock()
        self.last_request_time = 0.0

    def wait_if_needed(self) -> None:
        if self.rpm_limit <= 0:
            return
        
        spacing = 60.0 / self.rpm_limit
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < spacing:
                sleep_time = spacing - elapsed
                logger.debug(f"Rate limiter: sleeping for {sleep_time:.2f}s to respect RPM limit of {self.rpm_limit}")
                time.sleep(sleep_time)
            self.last_request_time = time.time()


# Instantiate a global rate limiter. Default to 100 RPM
_RPM_LIMIT = float(os.getenv("OPENROUTER_RPM_LIMIT", "100"))
rate_limiter = OpenRouterRateLimiter(rpm_limit=_RPM_LIMIT)


def generate_readme_md(clean_text: str) -> str:
    """
    Generate a clean, structured Markdown document from the cleaned README plain text
    using the OpenRouter API. Enforces rate limits and automatically retries with 
    exponential backoff on HTTP 429.
    """
    if not clean_text or not clean_text.strip():
        return ""

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL_ID") or "meta-llama/llama-3.3-70b-instruct"
    url = os.getenv("OPENROUTER_API_URL") or "https://openrouter.ai/api/v1/chat/completions"

    if not api_key:
        logger.warning(
            "No OPENROUTER_API_KEY found in the environment. "
            "Skipping README markdown generation."
        )
        return ""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("CLIENT_URL") or "http://localhost:8081", # Optional, for including your app on openrouter.ai rankings.
        "X-Title": "GH Social ML", # Optional. Shows in rankings on openrouter.ai.
    }
    
    prompt = (
        "You are an expert technical writer. Convert the following plain text version of a GitHub repository README "
        "into a clean, well-structured, and highly readable Markdown document. Use logical headings, subheadings, "
        "bullet points, and an improved overall structure.\n\n"
        "Instructions:\n"
        "- Do NOT add or introduce new facts, library features, or hallucinate information not present in the source text.\n"
        "- Preserve all important technical details, configurations, installation instructions, code snippets, and commands.\n"
        "- Remove unnecessary boilerplate or redundancies to make it clean and readable.\n"
        "- Return ONLY the generated Markdown text. Do NOT include any introductory or concluding comments, greetings, or conversational remarks.\n\n"
        f"Source text:\n{clean_text}"
    )

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

    max_retries = 3
    backoff_factor = 2.0

    for attempt in range(max_retries + 1):
        # Enforce RPM spacing before making the call
        rate_limiter.wait_if_needed()

        try:
            logger.info(f"Generating README markdown using OpenRouter model '{model}' (Attempt {attempt + 1}/{max_retries + 1})...")
            # Set a 45-second timeout to accommodate model reasoning/latency
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            
            if response.status_code == 200:
                res_json = response.json()
                try:
                    text_out = res_json["choices"][0]["message"]["content"].strip()
                    # Clean up potential markdown code fences wrapping the response
                    if text_out.startswith("```"):
                        text_out = re.sub(r"^```[a-zA-Z]*\n?", "", text_out)
                        if text_out.endswith("```"):
                            text_out = text_out[:-3].strip()
                    return text_out.strip()
                except (KeyError, IndexError) as e:
                    logger.error(f"Failed to parse OpenRouter API response payload: {e}")
                    return ""
            
            elif response.status_code == 429:
                if attempt < max_retries:
                    sleep_time = backoff_factor ** attempt * 2.0
                    logger.warning(f"OpenRouter API rate limit exceeded (HTTP 429). Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error("OpenRouter API rate limit exceeded (HTTP 429) and max retries exhausted.")
                    return ""
            
            else:
                logger.error(f"OpenRouter API returned error status {response.status_code}: {response.text}")
                return ""
                
        except Exception as exc:
            if attempt < max_retries:
                logger.warning(f"Error calling OpenRouter API: {exc}. Retrying...")
                time.sleep(1.0)
                continue
            else:
                logger.error(f"Error calling OpenRouter API after max retries: {exc}")
                return ""
    
    return ""

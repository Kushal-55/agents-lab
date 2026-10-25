import os
import logging
from typing import Optional
from openai import OpenAI, APIError
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


# Configure logging for system events (keeps UI clean)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# OpenRouter offers several truly free endpoints. Other great free options:
DEFAULT_MODEL = "google/gemini-2.5-flash"

def query_free_llm(prompt: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    """
    Sends a query to a free-tier LLM API using the standard OpenAI SDK.
    
    Args:
        prompt (str): The user's input query.
        model (str): The model identifier with the ':free' tag.
        
    Returns:
        Optional[str]: The model's text response, or None if the call fails.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("Missing OPENROUTER_API_KEY environment variable.")
        return None

    try:
        # We use the official OpenAI SDK but override the base_url 
        # to point to our free-tier provider
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        logger.info(f"Sending request to {model}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except APIError as e:
        logger.error(f"API Provider Error: {e.message}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        
    return None

def main() -> None:
    """Runs an interactive CLI chat loop."""
    print(f"LLM chat ({DEFAULT_MODEL})")
    print("Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
                
            output = query_free_llm(query)
            
            if output:
                print(f"\nAI:\n{output}")
            else:
                print("\nFailed to get a response. Check the log output.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
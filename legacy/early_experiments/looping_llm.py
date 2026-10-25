import os
import logging
from typing import Optional
from openai import OpenAI, APIError
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Configure logging for system events (keeps UI clean)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemini-2.5-flash"

def query_free_llm(client: OpenAI, prompt: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    """
    Sends a query to an LLM API using the standard OpenAI SDK.
    
    Args:
        client (OpenAI): The initialized OpenAI client.
        prompt (str): The user's input query.
        model (str): The model identifier.
        
    Returns:
        Optional[str]: The model's text response, or None if the call fails.
    """
    try:
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
    """Initializes the client and runs an interactive CLI chat loop."""
    
    # 1. Verify API Key is loaded
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("Missing OPENROUTER_API_KEY. Please ensure it is set in your .env file.")
        return

    # 2. Initialize the client ONCE for the entire session
    try:
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return

    print(f"OpenRouter chat ({DEFAULT_MODEL})")
    print("Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nYou: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not query:
                continue
                
            # 3. Pass the initialized client to the query function
            output = query_free_llm(openrouter_client, query)
            
            if output:
                print(f"\nAI:\n{output}")
            else:
                print("\nFailed to get a response. Check the log output.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
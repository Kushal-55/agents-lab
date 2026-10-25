import os
import yfinance as yf
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables (API Keys)
load_dotenv()

# 1. Initialize the Free LLM via OpenRouter
# Note: LangChain's ChatOpenAI client works perfectly with OpenRouter by overriding the base_url.
llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.2 # Low temperature for more analytical, deterministic financial answers
)

# ==========================================
# 2. Define Complex Financial Tools
# ==========================================

@tool
def get_stock_info(ticker: str) -> str:
    """Gets real-time price, market cap, and key valuation metrics for a given stock ticker."""
    print(f"[tool]Fetching live data for {ticker.upper()}...")
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        price = info.get('currentPrice', info.get('regularMarketPrice'))
        if not price:
            return f"Could not fetch price data for {ticker}. The ticker might be invalid."
            
        market_cap = info.get('marketCap', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 'N/A')

        # Yahoo returns the string 'N/A' (from our default) when a field is missing, and the
        # number format below only works on numbers, so format conditionally.
        market_cap_text = (f"${market_cap:,.0f}" if isinstance(market_cap, (int, float))
                           else market_cap)

        return (f"Ticker: {ticker.upper()}\n"
                f"Current Price: ${price:,.2f}\n"
                f"52-Week High: ${fifty_two_week_high}\n"
                f"Market Cap: {market_cap_text}\n"
                f"P/E Ratio: {pe_ratio}")
    except Exception as e:
        return f"Error fetching market data for {ticker}: {e}"

@tool
def get_financial_news(ticker: str) -> str:
    """Fetches the latest financial news headlines for a given stock ticker."""
    print(f"[tool]Fetching news for {ticker.upper()}...")
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:3] # Retrieve the top 3 most recent articles
        
        if not news:
            return f"No recent news found for {ticker}."
        
        headlines = [f"- {n['title']} (Source: {n['publisher']})" for n in news]
        return "\n".join(headlines)
    except Exception as e:
        return f"Error fetching news for {ticker}: {e}"

@tool
def calculate_loan_payment(principal: float, annual_rate: float, years: int) -> str:
    """Calculates the fixed monthly payment for a mortgage or standard loan."""
    print(f"[tool]Calculating loan amortization...")
    if annual_rate == 0:
        return f"Monthly payment (0% interest): ${principal / (years * 12):,.2f}"
    
    monthly_rate = (annual_rate / 100) / 12
    num_payments = years * 12
    payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    
    return (f"Loan Details: ${principal:,.2f} at {annual_rate}% for {years} years.\n"
            f"Monthly Payment: ${payment:,.2f}")

@tool
def calculate_compound_interest(principal: float, annual_rate: float, years: int, monthly_contribution: float = 0.0) -> str:
    """Calculates the future value of an investment with compound interest and optional monthly contributions."""
    print(f"[tool]Calculating compound growth...")
    rate = annual_rate / 100
    months = years * 12
    monthly_rate = rate / 12
    
    # Compound the initial principal
    future_value_principal = principal * (1 + rate)**years
    
    # Compound the recurring monthly contributions
    if monthly_contribution > 0 and monthly_rate > 0:
        future_value_contributions = monthly_contribution * (((1 + monthly_rate)**months - 1) / monthly_rate)
    else:
        future_value_contributions = monthly_contribution * months
        
    total_future_value = future_value_principal + future_value_contributions
    total_contributed = principal + (monthly_contribution * months)
    total_interest = total_future_value - total_contributed
    
    return (f"Forecast after {years} years at {annual_rate}% annual return:\n"
            f"Total Future Value: ${total_future_value:,.2f}\n"
            f"Total Principal Contributed: ${total_contributed:,.2f}\n"
            f"Total Interest Earned: ${total_interest:,.2f}")

# ==========================================
# 3. Compile the Agent and Memory Layer
# ==========================================

tools = [get_stock_info, get_financial_news, calculate_loan_payment, calculate_compound_interest]

system_prompt = (
    "You are an expert quantitative Financial Assistant. "
    "You help users analyze markets, check real-time stock prices, look up financial news, and model investments or loans. "
    "Always rely on your tools to retrieve accurate market data or perform financial math; never hallucinate or guess stock prices. "
    "When explaining financial concepts or tool results, be clear, professional, and structured."
)

# LangChain's modern create_agent handles the routing loop automatically
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

def main():
    print("Financial assistant ready.")
    print("It can look up stock quotes and news, and calculate loan payments and compound growth.")
    print("Type 'quit' to exit.")
    print("-" * 80)
    
    # The thread_id is set once, outside the loop. LangGraph's checkpointer keys the saved
    # conversation on it, so reusing the same id keeps every turn in one conversation.
    config = {"configurable": {"thread_id": "session_001"}}
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye.")
                break
            if not user_input:
                continue
                
            print("Thinking...")

            # The agent calls tools as needed and stores the exchange under this thread_id.
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config
            )
            
            # Extract the final textual response from the agent
            final_message = response["messages"][-1].content
            print(f"\nAgent:\n{final_message}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
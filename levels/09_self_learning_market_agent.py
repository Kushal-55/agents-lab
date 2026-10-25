"""LEVEL 9 (capstone) - A self-learning market agent.

Concept:   An ensemble of forecasting "experts" (trend, momentum, mean-reversion, base rate and
           a from-scratch logistic regression). The agent
             1. forecasts P(price higher in N days) and writes it in a JOURNAL,
             2. later scores each forecast against what really happened,
             3. updates how much it trusts each expert (Hedge / multiplicative weights),
             4. records "lessons", and reports its track record next to naive baselines.
           An LLM sits on top (the `chat` command) to explain results using these tools.
Concepts to learn: online learning, leak-free walk-forward evaluation, probabilistic scoring
           (Brier), naive baselines, and why "self-learning" is not the same as "can beat the
           market".

No LLM or API key is needed for the learning engine itself:
  python levels/09_self_learning_market_agent.py demo                  # 5-second offline demo
  python levels/09_self_learning_market_agent.py backtest SPY          # walk-forward backtest
  python levels/09_self_learning_market_agent.py forecast AAPL         # forecast + journal it
  python levels/09_self_learning_market_agent.py resolve               # run daily/weekly: learn
  python levels/09_self_learning_market_agent.py report                # live track record
  python levels/09_self_learning_market_agent.py chat                  # talk to the LLM analyst

Educational only. Short-horizon forecasts are barely better than chance. Not financial advice.
"""

from agentlab.finance.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

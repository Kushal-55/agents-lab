# Agents Lab

A set of small, runnable programs that build up, one step at a time, from a single call to a
language model to an agent that makes market forecasts, keeps a record of them, checks them
against what actually happened, and adjusts how much it trusts each of its own methods.

Everything runs on free services: a model running locally through Ollama, or the free tiers of
OpenRouter, Groq or Gemini. There are no paid APIs.

> This is a learning project, not financial advice. Short-horizon market forecasts are barely
> better than chance, and the program says so in its own output.

## Purpose

I wanted to understand how AI agents work by building each piece myself instead of starting
from a framework. Each program (a "level") adds exactly one idea to the one before it and
stops at the point where that idea runs out, which is the reason the next level exists:

1. one request to a model
2. a conversation (the model is stateless, so history has to be resent)
3. tools, using a hand-made JSON protocol
4. tools again, using the provider's function calling, inside a bounded reasoning loop
5. memory that survives restarts
6. retrieval over my own documents, with citations
7. several cooperating roles (planner, researchers, writer, critic)
8. live market data as tools
9. a forecaster that learns from its own results

## What I am trying to achieve

* Understand the mechanics of agents well enough to explain and debug them: the message
  format, why tool calls must stay paired with their results, why loops need a step limit,
  what memory and retrieval actually are.
* Treat everything a model or the internet hands back as untrusted input: validate tool
  arguments against a schema, never `eval` model output, label news headlines as quoted data.
* Build a forecaster whose evaluation I can trust. Every forecast is written down before the
  outcome is known, backtests only use information that existed on each day, and every result
  is shown next to simple baselines such as "always guess the common outcome".
* Learn the difference between a system that *learns* and a system that *beats the market*.
  The capstone learns correctly, and on real data it shows no edge over the baselines. That
  result is expected, and being able to measure it properly is the point.

## Project layout

```
levels/            the nine runnable programs, in order
src/agentlab/      shared library used by the levels
  finance/         market data, features, forecasting experts, journal, learner, CLI
data/knowledge/    small finance notes searched by the retrieval levels (add your own .md/.txt)
data/runtime/      journal, memory, price cache and logs (created on first run, git-ignored)
tests/             offline test suite
docs/              design notes (ARCHITECTURE.md)
legacy/            my first experiments, kept for reference
```

## What each part does

### The levels (`levels/`)

| Level | File | What it does | Why the next level exists |
|---|---|---|---|
| 1 | `01_llm_call.py` | Sends one prompt to the model and prints the answer. Shows the request shape (model, messages, max tokens) and how failures are handled. | It forgets everything between calls. |
| 2 | `02_conversation_loop.py` | A chat loop. Keeps the message history and resends it each turn, trimming old messages to bound cost. `/reset` clears it. | It can only talk; it can't calculate or look anything up. |
| 3 | `03_tools_from_scratch.py` | Tool use with a protocol I invented: the model replies with JSON naming a tool, my code runs it and feeds the result back. | Parsing free text is fragile (extra prose, broken JSON, made-up tool names). |
| 4 | `04_react_agent.py` | Native function calling in a reason, act, observe loop with a calculator, clock, loan and compound-growth tools. Arguments are validated and the loop is capped at `max_steps`. | Every run starts blank. |
| 5 | `05_memory_agent.py` | Adds long-term memory: `remember` and `recall` tools backed by SQLite. Text that looks like a password or API key is refused. | It only knows what I told it. |
| 6 | `06_rag_agent.py` | Retrieval-augmented generation over `data/knowledge/` using BM25 I wrote myself. Answers cite the source file and the agent says it doesn't know when the notes don't cover a question. | It has no live data. |
| 7 | `07_multi_agent.py` | A pipeline of roles built from one model: a planner splits the question, researchers answer each part with tools, a writer drafts, a critic checks the draft against the notes and can force one revision. | It still has no market data and doesn't learn. |
| 8 | `08_finance_analyst.py` | An analyst agent with price summaries, technical indicators and news from Yahoo Finance (cached on disk), plus everything from earlier levels. `--synthetic` uses fake offline prices. | It reports facts but doesn't forecast or learn. |
| 9 | `09_self_learning_market_agent.py` | The capstone. See below. | This is the end of the path. |

### The capstone (level 9)

Five "experts" each give a probability that a price will be higher in N trading days: a trend
rule, a momentum rule, a mean-reversion rule, the historical base rate, and a logistic
regression I wrote with NumPy. An ensemble averages them using weights that it learns.

```
prices -> experts (each gives P(up in N days)) -> weighted average -> forecast
                                                                          |
                                      written to the journal (SQLite) <---+
                                                                          |
                              N trading days pass, real prices arrive     |
                                                                          v
              resolve: score the forecast, update the weights (Hedge algorithm)
```

After each outcome, `w_i <- w_i * exp(-eta * (p_i - outcome)^2)`, then the weights are
renormalised with a small floor so no expert is ever dropped completely. The first time a
ticker is forecast, the weights are warm-started by replaying its price history through the
same loop.

The learning engine needs no LLM and no API key. The model is only used by `chat`, to explain
results in plain language.

```bash
python levels/09_self_learning_market_agent.py demo            # 5-second offline demonstration
python levels/09_self_learning_market_agent.py backtest SPY    # walk-forward test on real prices
python levels/09_self_learning_market_agent.py forecast AAPL   # forecast and record it in the journal
python levels/09_self_learning_market_agent.py resolve         # run later (daily/weekly): score and learn
python levels/09_self_learning_market_agent.py report          # live track record
python levels/09_self_learning_market_agent.py lessons         # summaries written during scoring
python levels/09_self_learning_market_agent.py chat            # talk to the LLM analyst
```

Add `--synthetic` before the command to use fake prices and a separate journal.

### Shared library (`src/agentlab/`)

| Module | What it does |
|---|---|
| `config.py` | Reads settings from environment variables / `.env`, validates them, keeps the API key out of `repr`. |
| `bootstrap.py` | Start-up shared by the level scripts: load settings, set up logging, build the LLM client. |
| `log.py` | Logging setup (console plus rotating file) with a filter that redacts anything shaped like an API key. |
| `llm.py` | One client for Ollama, OpenRouter, Groq and Gemini (all speak the OpenAI protocol). Turns SDK errors into readable messages. |
| `tools.py` | `Tool` and `ToolRegistry`. Validates the model's arguments against a JSON schema, contains crashes, caps output size. |
| `builtin_tools.py` | Calculator, clock, loan payment, compound growth, memory tools and knowledge-search tool. |
| `safe_math.py` | The calculator's evaluator. Parses to an AST and allows only arithmetic, so there is no `eval`. |
| `agent.py` | The agent loop: call the model, run requested tools, repeat until done or `max_steps`. |
| `memory.py` | `ConversationMemory` (rolling window of whole turns) and `FactStore` (durable facts in SQLite). |
| `retrieval.py` | Chunking and a BM25 index; only reads `.md`/`.txt` files directly inside the knowledge folder. |
| `multi_agent.py` | The planner / researcher / writer / critic pipeline used by level 7. |
| `jsonutil.py` | Pulls a JSON object out of messy model output. |
| `repl.py` | The interactive loop shared by levels 4 to 8. |
| `db.py` | Small SQLite helper that always commits or rolls back and closes the connection. |

### Market code (`src/agentlab/finance/`)

| Module | What it does |
|---|---|
| `data.py` | Price providers: Yahoo Finance, an on-disk cache with stale fallback, seeded fake prices, and a "time machine" that reveals history a day at a time for demos. Also validates tickers and periods. |
| `features.py` | Indicators (returns, moving averages, RSI, MACD, Bollinger position, volatility, volume) and the forward-looking labels, kept separate so the features never see the future. |
| `experts.py` | The rule-based experts, the base rate, the walk-forward logistic expert, and the `Ensemble` with the Hedge update. |
| `model.py` | Logistic regression from scratch (standardise, gradient descent, L2). |
| `backtest.py` | Walk-forward replay: on each day, learn from the outcome that just became known, then forecast. |
| `evaluation.py` | Accuracy, Brier score and Brier skill against the base rate, and a z-score adjusted for overlapping samples, with a plain verdict. |
| `journal.py` | SQLite journal of forecasts, outcomes, weights and lessons. |
| `learner.py` | `Forecaster`: forecast, backtest and resolve. |
| `report.py` | Text formatting for forecasts, backtests and the track record. |
| `analysis.py` | Price summary and technical snapshot text for the level 8 tools. |
| `news.py` | Headline fetching with sanitising. |
| `tools.py` | The market and forecasting functions exposed to the LLM as tools. |
| `assembly.py` | Builds the provider, journal, tools and analyst agent. |
| `cli.py` | The command-line interface for level 9. |

### Everything else

* `tests/` has offline tests with a scripted fake LLM and synthetic prices. They cover the tool
  registry, the agent loop, memory, retrieval, the calculator, the CLI, and, importantly, that
  the features, the base rate and the replay don't look ahead.
* `docs/ARCHITECTURE.md` explains the design decisions.
* `data/knowledge/` holds five short finance notes used by the retrieval levels.
* `legacy/early_experiments/` has my first LangChain and OpenRouter scripts (a single call, a
  chat loop, and a small finance agent). They aren't part of the path above and need the
  packages in `legacy/requirements-legacy.txt`.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then edit .env; it is git-ignored
```

### Choosing a free model (only needed for the chat levels)

| Provider | Cost | Setup |
|---|---|---|
| Ollama (default) | Free, runs locally | Install from [ollama.com](https://ollama.com), then `ollama pull llama3.1:8b` |
| OpenRouter | Free models (names end in `:free`) | Key from openrouter.ai; in `.env` set `LLM_PROVIDER=openrouter` and `OPENROUTER_API_KEY` |
| Groq | Free tier | `LLM_PROVIDER=groq`, `GROQ_API_KEY` |
| Gemini | Free tier | `LLM_PROVIDER=gemini`, `GEMINI_API_KEY` |

Free tiers get rate limited and model names change; set `LLM_MODEL=<name>` to override the
default. Levels 4 to 9 need a model that supports function calling.

```bash
python levels/01_llm_call.py "Explain compound interest in one sentence"
python levels/04_react_agent.py
python levels/06_rag_agent.py
```

## Example results

Offline demo (fake prices with a hidden slow trend, starting from equal weights). After 120
simulated days of forecast, wait, score, learn:

```
expert            start weight  end weight
trend                      20%          4%
momentum                   20%          3%
mean_reversion             20%          1%
base_rate                  20%         46%
ml_logreg                  20%         47%
```

Real data, `backtest SPY --horizon 5`, 951 days, each forecast using only what was known that
day (numbers change as new prices arrive):

```
expert           accuracy   naive   skill      z   verdict
trend               55.4%   61.8%  -0.016  -1.83   no demonstrable edge over naive baselines
ml_logreg           57.3%   61.8%  -0.055  -1.29   no demonstrable edge over naive baselines
ensemble            55.2%   61.8%  -0.023  -1.89   no demonstrable edge over naive baselines
```

SPY went up in most 5-day windows, so "always guess up" scores 61.8%, and none of the methods
beats that. Liquid markets are close to efficient, so this is the expected outcome. What the
project demonstrates is the machinery for testing an idea properly.

## Logging

The standard `logging` module is used throughout. The console shows WARNING and above
(set `LOG_LEVEL=INFO` or `DEBUG` for more). A rotating log file in `data/runtime/agentlab.log`
records INFO and above. Anything shaped like an API key is redacted before it is written, and
user text and model output are logged by length rather than content. Interactive output
(prompts, answers, reports) uses `print`; diagnostics never do.

## Tests and lint

```bash
pytest          # 167 tests, no network or API key needed
ruff check .
```

The tests include a security scan that fails if a key-shaped string shows up anywhere in the
source, docs or config. See [SECURITY.md](SECURITY.md) for how secrets are handled.

## Limitations

* Daily bars, price-only features, no trading costs, no news model.
* The Hedge weights follow recent performance, so one expert can dominate for a while even if
  its long-run record is poor. In the real-data backtest above, mean reversion ends with most
  of the weight despite being the worst expert overall.
* `yfinance` is an unofficial free API and can rate-limit or change. Results are cached for
  six hours, and a stale cache is used (with a warning) when the network is down.
* Synthetic data is for tests and demos, not for conclusions.

## Ideas for later

Add a volatility-regime expert, try a different loss or learning rate, add a news sentiment
feature, forecast returns instead of direction, or swap BM25 for local embeddings. Running
`resolve` regularly makes the live track record grow.

## Credits

The early levels follow the arc of the *ML105: Agentic AI* mini course by Rola Dali. The code
here is my own implementation using free services instead of paid cloud APIs.

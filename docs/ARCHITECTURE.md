# Architecture and design notes

## 1. The agent loop (levels 4 and up)

```
user -> Agent.run -> LLM --(no tool calls)--> final answer
                      ^  |
                      |  +-(tool calls)-> ToolRegistry.execute -> validate args -> run tool
                      |                          |                                  |
                      +-- tool result (size-capped string, errors included) <-------+
              bounded by max_steps; if the budget runs out, one last call is made with tools off
```

* `llm.py`: one OpenAI-compatible client covers Ollama, OpenRouter, Groq and Gemini.
  `ChatModel` is a `Protocol`, so the tests pass in a scripted `FakeLLM` and never touch the
  network.
* `tools.py`: a `Tool` is a name, a description, a JSON schema and a function. The registry
  treats model output as untrusted: it validates arguments against the schema, rejects unknown
  arguments, contains crashes, and truncates long output.
* `memory.py`: `ConversationMemory` trims whole turns, because an assistant message with tool
  calls must stay next to its tool replies or providers reject the history. `FactStore` is
  durable SQLite memory that refuses text shaped like a secret.
* `retrieval.py`: BM25 from scratch. No embedding API is needed, and for a handful of short
  documents it is a good baseline.

## 2. The learning loop (level 9)

1. **Experts** (`finance/experts.py`): each turns today's features into P(up in N days). The
   rule experts are deliberately modest (at most 0.10 away from 0.5) because real signals are
   weak.
2. **Ensemble**: a weighted average. Weights are updated with **Hedge** using the Brier loss,
   plus a **fixed-share** floor (`eta=3`, `alpha=0.01`, chosen with a small grid search on
   synthetic data; see `experts.py`).
3. **Journal** (`finance/journal.py`): every forecast is stored before the outcome is known.
   Marking a forecast resolved and saving the new weights is one transaction, and a forecast
   can only be learned from once.
4. **Resolve** (`finance/learner.py`): fetch fresh prices, find the outcome of each matured
   forecast, update the weights in date order, and write a summary lesson every five scored
   forecasts.
5. **Replay** (`finance/backtest.py`): the same loop run over history. It is used both to
   warm-start the weights and as the backtest.

### Why the evaluation can be trusted

| Risk | What prevents it | Test |
|---|---|---|
| Features see the future | Only trailing windows; labels are computed separately | `test_features_have_no_lookahead` |
| The model trains on labels not yet known | The model for day t uses only rows with `i + horizon <= t` | `test_replay_has_no_lookahead` |
| The base rate peeks ahead | Expanding mean of labels shifted by the horizon | `test_base_rate_uses_only_labels_known_at_the_time` |
| Overlapping samples overstate significance | The z-score uses `n / horizon` effective samples | `test_evaluate_metrics_and_verdicts` |
| Comparing against nothing | Naive-baseline accuracy and Brier skill are always reported | `format_forecast` |

The look-ahead tests recompute results on truncated history and require identical output. If a
leak is introduced (for example by using `shift(-1)` in a feature), they should fail.

### Known limitations

* Hedge weights follow recent performance, so one expert can dominate for a stretch even when
  its full-period record is poor. On real data the ensemble does not beat the base rate.
* Daily bars, price-only features, no trading costs, no news model. Synthetic data is for
  testing, not for drawing conclusions.
* `yfinance` is an unofficial free API and can rate-limit or change. Results are cached for six
  hours, and a stale cache is used (with a warning) when the network is down.

## 3. Security

* Secrets live only in `.env` (git-ignored) or the environment. `Settings.api_key` is excluded
  from `repr`, the logging filter redacts key-shaped strings, and `FactStore` refuses to save
  them.
* Everything that comes from the model or the user is validated before use: tickers, periods,
  horizons, JSON arguments, expression length and complexity, file types and locations.
* External text (news) is capped, stripped of control characters and labelled as untrusted in
  the tool output and in the system prompt. This reduces, but does not remove, the risk of
  prompt injection.
* The library has no `eval`, `exec`, `pickle`, `shell=True` or SQL built from user input.

## 4. Extending it

* **A tool**: create `Tool(name, description, object_schema(...), func)`, raise `ToolError` for
  expected failures, and register it. See `builtin_tools.py`.
* **A rule-based expert**: add a column to `rule_expert_probs` (P(up) per row, built from
  `FEATURE_COLUMNS`) and its name to `RULE_EXPERTS`. Replay, forecasting, the journal and the
  reports then pick it up. A learned expert like `ml_logreg` also needs a branch in `replay`
  and in `Forecaster.forecast`.
* **A data source**: implement `PriceProvider.fetch(ticker, period) -> DataFrame` (see
  `data.py`).

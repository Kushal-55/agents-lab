# Security notes

This is a personal learning project, but it handles API keys, a prediction journal and
personal notes, so it follows a few rules.

## Handling secrets

* Keys live only in `.env` (or real environment variables). `.env` is git-ignored and should be
  mode 600 (`chmod 600 .env`). `.env.example` must never contain a real value.
* Never paste a key into source, docs, tests, notebooks or a prompt. Test fixtures build fake
  keys at runtime (`tests/fake_secrets.py`) so no key-shaped literal sits in the repo.
* `pytest` includes a repo scan (`tests/test_security.py`) that fails if a key-shaped string
  appears in any source, doc or config file. Run it before sharing or committing.
* If a key might have been exposed (shared folder, zip, screenshot, synced drive, pasted into
  a chat), revoke it at the provider and create a new one. Deleting the text is not enough.
* When sharing the project, send the source only: leave out `.env`, `venv/`, `data/runtime/`
  and any cache folders.

## What the code does to limit damage

* `Settings.api_key` is excluded from `repr`, so it can't leak through logs or tracebacks.
* All log output passes through a redactor (`agentlab/log.py`) that hides key-shaped strings,
  bearer tokens, JWTs, private keys, URL credentials and `password=...` style assignments.
  It runs on the final formatted text, so secrets inside exception tracebacks are caught too.
* Logs record tool names, counts and sizes, not what the user typed or what the model sent.
* Log files, the journal and the memory database are created readable only by the owner.
* `FactStore` (long-term memory) refuses text that looks like a credential.
* The provider URL comes from a fixed table, not from an environment variable, so a changed
  `.env` can't redirect your API key to another host.
* Model output is untrusted: tool arguments are validated against a JSON schema, the
  calculator parses an AST instead of calling `eval`, news headlines are sanitised and labelled
  as quoted data, tickers and periods are validated against whitelists, SQL is parameterised,
  and the document index refuses symlinks and files outside its folder.

## Known limitations

* The journal and memory databases are plain SQLite files. Permissions protect them from other
  users on the machine, but they are not encrypted.
* A prompt-injected model could ask `remember` to store misleading text. Facts are short and
  credential-shaped text is refused, but stored facts are trusted when recalled later.
* Dependencies use lower bounds only (no lock file), so installs pick up newer versions.
  Run `pip-audit` now and then.

## Audit log

* 2026-10-25: full review of code, config, logs, databases and caches for secrets and personal
  data; dependency scan. Fixes: tracebacks are now redacted, log files and databases are
  owner-only, logs no longer record user text, more redaction patterns, a repo scan test, extra
  ignore rules, and `certifi` and `pip` upgraded in the venv.

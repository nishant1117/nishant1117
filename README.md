# Jira RAG Agent

A responsive, Claude-powered RAG agent for Jira epics. It exposes two tools
that Claude can call based on free-form user input:

- **`analyzer`** - pulls a Jira epic and all its tickets, indexes them into a
  vector store, and uses Claude to generate test cases, edge cases,
  regression analysis, bug detection, detailed analysis, insights, or an
  action plan.
- **`comparator`** - runs the analyzer on two epics and produces a delta
  analysis: scope differences, risk deltas, shared dependencies, and
  recommended actions.

The `agent_cli.py` entry point wires the tools into a Claude tool-use loop so
users can just type what they want (`"compare PROJ-100 and PROJ-200 on
regression risk"`) and Claude decides which tool to invoke. Turn-to-turn
results are cached so follow-ups are cheap.

## Layout

| File | Purpose |
| --- | --- |
| `config.py` | Loads env vars (Jira creds, Anthropic key, RAG tunables). |
| `jira_client.py` | Fetches epic, tickets, subtasks, links, comments, changelog. |
| `vector_store.py` | ChromaDB + `RecursiveCharacterTextSplitter` wrapper. |
| `test_generator.py` | All Claude prompting (analyzer + comparator logic). |
| `rag_agent.py` | Orchestrator: Jira -> vector store -> Claude -> results. |
| `api.py` | FastAPI server exposing the same capabilities as HTTP endpoints. |
| `agent_cli.py` | Interactive Claude tool-use CLI (the responsive front-end). |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in JIRA_* and ANTHROPIC_API_KEY
```

## Use it

Interactive (recommended):

```bash
python agent_cli.py
you> Analyze epic PROJ-123 and list the top regression risks
you> Now compare it with PROJ-200 on scope overlap
```

Single prompt:

```bash
python agent_cli.py --prompt "Analyze PROJ-123 and summarize critical bugs"
```

Or run the FastAPI server:

```bash
uvicorn api:app --reload --port 8000
```

## How the tool-use loop works

1. User types a natural-language request.
2. Claude (system prompt in `agent_cli.py`) sees both tool schemas and
   decides whether to call `analyzer` or `comparator` — and with which
   `focus` — based on the intent.
3. The CLI dispatches the call to `JiraRAGAgent`, which runs the RAG
   pipeline (Jira fetch -> Chroma index -> per-ticket Claude calls).
4. The tool result is fed back to Claude as a `tool_result` block.
5. Claude writes a grounded, final answer citing the real ticket keys,
   bug ids, and risk levels from the tool output.
6. Subsequent turns reuse the in-memory cache so follow-ups are fast.

## Environment variables

See `.env.example`. Key ones:

| Var | Notes |
| --- | --- |
| `JIRA_HOST` / `JIRA_EMAIL` / `JIRA_API_TOKEN` | Jira Cloud credentials. |
| `ANTHROPIC_API_KEY` | Anthropic API key. |
| `CLAUDE_MODEL` | Defaults to `claude-sonnet-4-5`. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `TOP_K_RETRIEVAL` | RAG tunables. |
| `MIN_TEST_CASES_PER_TICKET` / `MAX_TEST_CASES_PER_TICKET` | Test-gen bounds. |

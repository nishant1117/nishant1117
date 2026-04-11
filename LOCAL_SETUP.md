# Running this project locally in VS Code

## 1. Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended) on your `PATH`
- **VS Code** with the Python extension (it'll be suggested automatically the
  first time you open the folder — see `.vscode/extensions.json`)
- A **Jira Cloud API token** (Atlassian account → *Security* → *API tokens*)
- An **Anthropic API key** (<https://console.anthropic.com/>)

## 2. Get the code

```bash
git clone <your-remote-url> jira-rag-agent
cd jira-rag-agent
code .
```

When VS Code prompts you to *"Install recommended extensions"*, click **Install**.

## 3. Configure secrets

```bash
cp .env.example .env
```

Then open `.env` and fill in:

```
JIRA_HOST=https://your-domain.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=...              # from Atlassian
ANTHROPIC_API_KEY=sk-ant-...    # from console.anthropic.com
CLAUDE_MODEL=claude-sonnet-4-5
```

> `.env` is gitignored — never commit it.

## 4. Create the virtualenv and install dependencies

Either press **Ctrl/Cmd + Shift + P → "Tasks: Run Build Task"** (this runs
*"Python: install requirements"* which also creates `.venv`), or from the
integrated terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then **Ctrl/Cmd + Shift + P → "Python: Select Interpreter"** and pick
`./.venv/bin/python`. The debugger and terminal will now use that env.

## 5. Run the responsive Claude CLI

Press **F5** and pick **"Agent CLI (interactive)"** from the dropdown.
A terminal will open where you can just type:

```
you> Analyze epic PROJ-123 and list the top regression risks
you> Now compare PROJ-123 with PROJ-456 on scope overlap
```

Claude decides whether to call the `analyzer` or `comparator` tool based on
what you said, the CLI runs the RAG pipeline, and Claude writes back a
grounded answer. Follow-ups reuse the in-memory cache so they're fast.

For a single-prompt run, pick **"Agent CLI (one-shot prompt)"** from the
**F5** dropdown — VS Code will ask for the prompt string and print the reply.

## 6. Run the FastAPI server (optional)

Press **F5** and pick **"FastAPI (uvicorn reload)"**. The server starts at
<http://localhost:8000> with hot reload. Open `requests.http` and click
*"Send Request"* above any block to hit an endpoint — the REST Client
extension handles it.

Interactive API docs: <http://localhost:8000/docs>

## 7. What to try first

1. `you> Summarize epic PROJ-123` — fastest, small tool output.
2. `you> What critical bugs did you find in PROJ-123?` — Claude picks
   `analyzer` with `focus: bug_analysis`.
3. `you> Compare PROJ-123 and PROJ-456 on regression risk` — Claude picks
   `comparator`.
4. `you> Actually, focus on data integrity risks for PROJ-123` — follow-up
   reuses the cached epic, runs a custom question through the RAG context.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ANTHROPIC_API_KEY is not set` | You forgot step 3, or VS Code didn't load `.env`. Make sure `python.envFile` in `.vscode/settings.json` points at it (it does by default) and restart the debugger. |
| `chromadb` install fails on Windows | Install Microsoft C++ Build Tools, or use WSL. |
| Jira 401 / 403 | Your API token is wrong or your account lacks access to the epic. Verify by hitting `{JIRA_HOST}/rest/api/3/myself` with the same credentials. |
| Slow first run | Expected — the first epic pulls every ticket, subtask, comment, and changelog, then runs Claude per-ticket. Subsequent turns hit the in-memory cache. |
| `ModuleNotFoundError` | You're not inside the venv. Select the interpreter as described in step 4. |

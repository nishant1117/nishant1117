# How to run this in VS Code (zero coding required)

This guide assumes you've never written Python before. You'll end up with
a web app open in your browser that lets you analyze and compare Jira
epics using Claude. Total time: ~10 minutes on a new machine.

---

## ⚠️ Before you start — security

If you pasted API keys or tokens anywhere public (chat, email, pastebin,
a screenshot), **revoke them first** and generate new ones. Treat any
key that left your own computer as compromised.

- **Anthropic key**: <https://console.anthropic.com/settings/keys> → revoke → create new
- **Jira API token**: <https://id.atlassian.com/manage-profile/security/api-tokens> → revoke → create new

This project never stores your credentials in git. They go either into
your browser session, or into a local file (`.streamlit/secrets.toml`)
that is already listed in `.gitignore` and will not be pushed.

---

## Step 1 — Install the three things you need (one-time)

On your own computer, install these if you don't already have them. Pick
the installer for your operating system and accept all defaults unless
noted otherwise.

| Tool | Download | Note |
| --- | --- | --- |
| **VS Code** | <https://code.visualstudio.com/> | The editor. |
| **Python 3.11 or 3.12** | <https://www.python.org/downloads/> | **On Windows:** check ✅ *"Add python.exe to PATH"* on the first installer screen. |
| **Git** | <https://git-scm.com/downloads> | On macOS it's usually pre-installed. |

Once installed, close and reopen VS Code so it picks up Python.

---

## Step 2 — Get the code into VS Code

1. Open VS Code.
2. Close any open folder: **File → Close Folder** (skip if welcome page is already showing).
3. Press **Ctrl + Shift + P** (macOS: **Cmd + Shift + P**) to open the command palette.
4. Type **`Git: Clone`** and press Enter.
5. Paste this URL and press Enter:
   ```
   https://github.com/nishant1117/nishant1117.git
   ```
6. Pick any folder on your disk to save it in (e.g. `Documents`).
7. When VS Code asks *"Would you like to open the cloned repository?"*, click **Open**.
8. **Switch to the feature branch.** In the bottom-left corner of VS Code, you'll see the word **`main`** — click it. A dropdown opens at the top of the window. Pick **`origin/claude/jira-rag-agent-KTIvY`**. The files on the left should now include `streamlit_app.py`, `agent_cli.py`, `.vscode/`, etc.
9. A toast pops up: *"Do you want to install the recommended extensions?"* Click **Install**. This gets you the Python tooling.

---

## Step 3 — Install the project's dependencies (one-time)

1. Press **Ctrl + Shift + P** (macOS: **Cmd + Shift + P**).
2. Type **`Tasks: Run Build Task`** and press Enter. (Or press **Ctrl + Shift + B** / **Cmd + Shift + B** directly.)
3. A terminal panel opens at the bottom and starts installing packages. Wait until you see the prompt (`$` or `>`) come back. This takes 1–3 minutes. It's downloading Python libraries including `streamlit`, `chromadb`, `langchain`, and the Claude SDK.

If you see a red error like *"python3 not found"* on Windows, close VS Code, reopen it, and make sure Python is on your PATH (re-run the Python installer and tick the *"Add to PATH"* box).

---

## Step 4 — Pick the Python interpreter (one-time)

1. Press **Ctrl + Shift + P** / **Cmd + Shift + P**.
2. Type **`Python: Select Interpreter`** and press Enter.
3. Pick the one that ends in `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows). It's usually labeled `('.venv': venv)`.

The bottom-right corner of VS Code should now say something like `3.12.x ('.venv': venv)`.

---

## Step 5 — Run the web app

1. Press **F5**.
2. A small dropdown appears at the top of the screen listing debug configurations.
3. Pick **"Streamlit UI (recommended)"**.
4. A terminal opens at the bottom. After a few seconds you'll see:
   ```
   You can now view your Streamlit app in your browser.
   Local URL: http://localhost:8501
   ```
5. A browser tab opens automatically at <http://localhost:8501>. If it doesn't, click the `Local URL` link in the terminal.

---

## Step 6 — Enter your credentials (one-time)

The web app opens with an empty sidebar on the left. Fill in:

| Field | What to type |
| --- | --- |
| **Jira URL** | `https://healthtap.atlassian.net` |
| **Jira email** | `nishant.mahajan@healthtap.com` |
| **Jira API token** | your freshly-rotated Atlassian API token |
| **Anthropic API key** | your freshly-rotated Anthropic key |
| **Claude model** | leave as `claude-sonnet-4-5` |

Click **Save to disk**. A green *"Saved to `.streamlit/secrets.toml`. The file is gitignored."* message appears. Next time you run the app, the fields auto-fill from that file.

> **Why local-only:** `.streamlit/secrets.toml` is listed in `.gitignore`, so `git push` will never upload it. Your credentials stay on your laptop.

---

## Step 7 — Use it

The main area has three tabs:

### 💬 Chat
Type a natural-language request and press Enter. Claude decides whether
to run the analyzer (single epic) or the comparator (two epics) based
on what you said.

Try:
- `Analyze epic HT-1234 and list the top regression risks`
- `What critical bugs did you find in HT-1234?`
- `Compare HT-1234 and HT-1300 on scope overlap`

The first request to a given epic is slow (1–3 minutes) because the app
fetches every ticket, subtask, comment, and changelog from Jira, indexes
them, and runs Claude per-ticket. Follow-ups are fast because results
are cached.

### 🔍 Analyze Epic
A form with:
- **Jira epic key** — e.g. `HT-1234`
- **What do you want to see?** — drop-down with `summary`, `test_cases`,
  `edge_cases`, `regression`, `bug_analysis`, `insights`,
  `recommendations`, `action_plan`, or `full`.
- **Optional custom question** — e.g. *"Which tickets carry the highest
  data-integrity risk?"*

Click **Analyze**. Results render as expandable cards so you can drill
into individual test cases or edge cases.

### ⚖️ Compare Epics
Two epic keys and an optional goal. Click **Compare**. You get a
side-by-side summary and Claude's delta analysis.

---

## How to stop and restart

- **Stop the app**: click the red ⏹ square in the debug toolbar at the
  top of VS Code, or press **Shift + F5**.
- **Restart**: press **F5** → pick **"Streamlit UI (recommended)"** again.
- **Just reload after a code change**: Streamlit auto-reloads when you
  save a `.py` file, so you usually don't need to restart.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Sidebar says *"Could not initialise the agent"* | Credentials are wrong or Jira URL is wrong. Double-check the email, API token, and Jira URL in the sidebar, then click **Use for session**. |
| *Jira 401 / 403* | Your API token is invalid or your user can't see that epic. Test in your browser by visiting `https://healthtap.atlassian.net/rest/api/3/myself` — if that returns your details, the token works. |
| *"ANTHROPIC_API_KEY is not set"* | You haven't saved credentials yet. Fill in the sidebar and click **Use for session** or **Save to disk**. |
| `ModuleNotFoundError` | You skipped **Step 4** (pick interpreter). Do it now and rerun. |
| `pip install chromadb` fails on Windows | Install **Microsoft C++ Build Tools** (<https://visualstudio.microsoft.com/visual-cpp-build-tools/>), or use WSL2. The app will still run without `chromadb` — it just skips the RAG vector-store retrieval step and lets Claude work from the raw Jira data. |
| Browser tab never opens | Look in the VS Code terminal for `http://localhost:8501` and click it, or open the URL manually. |
| Streamlit complains about a port being in use | Stop the app (Shift + F5) and press F5 again. |
| *"First run takes forever"* | Expected. Epics with many tickets can take 1–3 minutes on the first pass because each ticket is individually analysed by Claude. The second time you ask about the same epic, it's instant. |

---

## What each file does (so you know where to look)

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | **The web UI you run.** All buttons, tabs, and chat live here. |
| `agent_cli.py` | Terminal-only version of the same tool-use loop. Used by the UI too. |
| `rag_agent.py` | Orchestrates fetching a Jira epic + running Claude per ticket. |
| `jira_client.py` | Wraps the Jira API (epics, tickets, comments, changelog). |
| `test_generator.py` | All the Claude prompts (test cases, bugs, comparisons, etc). |
| `vector_store.py` | Optional ChromaDB retrieval. Degrades gracefully if missing. |
| `config.py` | Reads env vars lazily so the UI can inject credentials. |
| `api.py` | Alternative FastAPI HTTP server (you don't need this for the UI). |
| `.vscode/launch.json` | F5 configs (Streamlit, CLI, FastAPI). |
| `.vscode/tasks.json` | Build tasks (create venv, install deps, run app). |
| `.streamlit/secrets.toml` | **Local only, gitignored.** Where your credentials are stored after you click *Save to disk*. |

---

## Golden path cheat sheet

```
1. Install VS Code + Python 3.12 + Git (one-time)
2. Ctrl+Shift+P → Git: Clone → paste repo URL
3. Bottom-left branch picker → origin/claude/jira-rag-agent-KTIvY
4. Install recommended extensions (toast)
5. Ctrl+Shift+P → Tasks: Run Build Task   (creates .venv + pip install)
6. Ctrl+Shift+P → Python: Select Interpreter → pick .venv
7. F5 → "Streamlit UI (recommended)"
8. Browser opens → fill sidebar → Save to disk
9. Chat tab → "Analyze HT-1234"
```

That's it. Everything after step 8 is reusable — you won't need to
re-enter credentials unless you hit **Revoke** in the Anthropic or Jira
consoles.

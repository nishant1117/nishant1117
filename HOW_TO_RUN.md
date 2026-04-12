# How to run this in VS Code (zero coding required)

Two ways to authenticate Claude are supported:

1. **Claude.ai subscription (recommended)** — uses your existing Pro/Max
   plan via the Claude Code CLI. No separate API bill.
2. **Anthropic API key** — direct per-token billing.

This guide walks through **Option 1** end-to-end. It's 10–15 minutes
on a fresh machine.

---

## ⚠️ Before you start — security

If you've pasted API keys or tokens anywhere public, **revoke them
first**. Any key that left your own computer should be treated as
compromised.

- **Anthropic key**: <https://console.anthropic.com/settings/keys>
- **Jira API token**: <https://id.atlassian.com/manage-profile/security/api-tokens>

This project never stores credentials in git. They go into your
browser session or into a local `.streamlit/secrets.toml` file that
is already listed in `.gitignore`.

---

## Step 1 — Install the tools you need (one-time)

| Tool | Download | Notes |
| --- | --- | --- |
| **VS Code** | <https://code.visualstudio.com/> | The editor. |
| **Python 3.11 or 3.12** | <https://www.python.org/downloads/> | **Windows:** tick ✅ *"Add python.exe to PATH"* on the installer. |
| **Git** | <https://git-scm.com/downloads> | macOS usually has it already. |
| **Node.js 18+** | <https://nodejs.org/> | Needed for Claude Code (subscription mode). Skip if using API-key mode only. |
| **Claude Code CLI** | run `npm install -g @anthropic-ai/claude-code` in a terminal once Node is installed | Needed for subscription mode. |

Close and reopen VS Code after installing so it picks up the new PATH.

---

## Step 2 — Log in to Claude Code (one-time, subscription mode only)

1. Open a terminal (any terminal — VS Code's built-in terminal is fine).
2. Run:
   ```
   claude /login
   ```
3. It opens a browser. Sign in with the same Claude.ai account you use
   to chat at claude.ai. Approve the connection.
4. The terminal prints something like `Logged in as you@example.com`.

From now on, every Claude API call from this project will count against
your Claude.ai plan instead of a separate API bill. You only need to do
this once per machine.

> **Plan check:** Claude Pro has tighter weekly limits than Max. A
> single epic analysis runs 5–10+ Claude calls, so if you're on Pro
> you may hit the cap quickly on a big epic. Max is recommended.

---

## Step 3 — Clone the repo into VS Code

1. Open VS Code.
2. **File → Close Folder** (skip if welcome page is already showing).
3. Press **Ctrl + Shift + P** (macOS: **Cmd + Shift + P**).
4. Type **`Git: Clone`** and press Enter.
5. Paste this URL and press Enter:
   ```
   https://github.com/nishant1117/nishant1117.git
   ```
6. Pick any folder on your disk.
7. Click **Open** when the *"Would you like to open the cloned repository?"* toast appears.
8. **Switch to the feature branch.** Click the word **`main`** in the bottom-left corner → dropdown at the top → pick **`origin/claude/jira-rag-agent-KTIvY`**. The file explorer should now show `streamlit_app.py`, `agent_cli.py`, `llm_client.py`, etc.
9. Accept the *"Install recommended extensions"* toast.

---

## Step 4 — Install Python dependencies (one-time)

1. Press **Ctrl + Shift + P** / **Cmd + Shift + P**.
2. Type **`Tasks: Run Build Task`** and press Enter. (Or press **Ctrl + Shift + B** / **Cmd + Shift + B**.)
3. Wait for pip install to finish (1–3 min). It downloads Streamlit, Jira SDK, Claude Agent SDK, Anthropic SDK, ChromaDB, and a few others.

---

## Step 5 — Pick the Python interpreter (one-time)

1. **Ctrl + Shift + P** / **Cmd + Shift + P** → **`Python: Select Interpreter`**.
2. Pick the entry labeled `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows).

Bottom-right of VS Code should now read something like `3.12.x ('.venv': venv)`.

---

## Step 6 — Run the web app

1. Press **F5**.
2. A dropdown appears at the top listing debug configurations.
3. Pick **"Streamlit UI (recommended)"**.
4. Wait a few seconds. A terminal opens and a browser tab appears at
   <http://localhost:8501>.

---

## Step 7 — Enter credentials (one-time per machine)

In the left sidebar:

1. **Claude authentication:** pick **"Claude.ai subscription (Claude Code)"**.
2. **Jira URL:** `https://healthtap.atlassian.net`
3. **Jira email:** `nishant.mahajan@healthtap.com`
4. **Jira API token:** your freshly-rotated Atlassian token.
5. **Claude model:** leave as `claude-sonnet-4-6`.

No Anthropic key is shown when subscription mode is selected — it uses
your `claude /login` session automatically.

Click **Save to disk**. Next time, the fields auto-fill from
`.streamlit/secrets.toml` (which is gitignored).

---

## Step 8 — Use it

Three tabs:

### 💬 Chat
Natural-language input. Claude decides whether to run the analyzer
(one epic) or the comparator (two epics).

Try:
- `Analyze epic HT-1234 and list the top regression risks`
- `What critical bugs did you find in HT-1234?`
- `Compare HT-1234 and HT-1300 on scope overlap`

First request to an epic is slow (1–3 minutes) because the app pulls
every ticket, subtask, comment, and changelog and runs Claude
per-ticket. Follow-ups are fast because results are cached in memory.

### 🔍 Analyze Epic
Form with epic key + focus dropdown + optional custom question. Click
**Analyze** to run.

### ⚖️ Compare Epics
Two epic keys + optional comparison goal. Click **Compare**.

---

## Stop / restart

- **Stop**: click the red ⏹ square in the debug toolbar, or **Shift + F5**.
- **Restart**: **F5** → **"Streamlit UI (recommended)"**.
- **Hot reload**: Streamlit auto-reloads when you save a `.py` file.

---

## Switching to API-key mode later

If you prefer to use an Anthropic API key instead of your
subscription:

1. In the sidebar, switch the **Claude authentication** radio to
   *"Anthropic API key (pay-per-token)"*.
2. Paste your key into the **Anthropic API key** field that appears.
3. Click **Use for session** or **Save to disk**.

That's it — everything else works the same.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| *"claude-agent-sdk not installed"* | The pip install didn't include it. Re-run the build task (Step 4) or run `pip install claude-agent-sdk` inside the venv. |
| *"Claude Agent SDK call failed: ... claude: command not found"* | Claude Code CLI isn't installed. Run `npm install -g @anthropic-ai/claude-code` in a terminal. |
| *"Please login first"* or *"not authenticated"* from the SDK | Run `claude /login` in a terminal and sign in with your Claude.ai account. |
| Sidebar says *"Could not initialise the agent"* | Jira credentials are wrong, or (if in API mode) your Anthropic key is invalid. Fix in the sidebar and click **Use for session**. |
| Jira 401 / 403 | Invalid token or no access to that epic. Test in your browser at `https://healthtap.atlassian.net/rest/api/3/myself`. |
| `pip install chromadb` fails on Windows | Install **Microsoft C++ Build Tools**, or use WSL2. The app still runs without ChromaDB — it just skips RAG retrieval. |
| Browser never opens | Look in the terminal for `http://localhost:8501` and click it manually. |
| Port 8501 in use | Stop the existing run (**Shift + F5**) and press F5 again. |
| First run takes forever | Expected — per-ticket Claude calls. Second run against the same epic is cached and fast. |
| *"Weekly usage limit reached"* in subscription mode | You're on Claude Pro and exhausted the weekly cap. Either upgrade to Max, or switch to API-key mode in the sidebar for the rest of the week. |

---

## What each file does

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | The web UI. Run this. |
| `llm_client.py` | **New.** Backend abstraction — picks API or Claude Code at runtime via `CLAUDE_AUTH_MODE`. |
| `agent_cli.py` | Terminal-only tool-use loop. Also powers the Chat tab. |
| `rag_agent.py` | Orchestrator: Jira → vector store → per-ticket Claude calls. |
| `jira_client.py` | Jira API wrapper. |
| `test_generator.py` | All Claude prompts (test cases, bugs, comparisons...). Now goes through `llm_client`. |
| `vector_store.py` | Optional ChromaDB retrieval. Degrades gracefully. |
| `config.py` | Lazy env-var reader. |
| `api.py` | Alternative FastAPI server (not needed for the UI). |
| `.vscode/launch.json` | F5 configs. |
| `.vscode/tasks.json` | Build tasks. |
| `.streamlit/secrets.toml` | **Local only, gitignored.** Where your saved credentials live. |

---

## Golden path cheat sheet (subscription mode)

```
1. Install VS Code + Python 3.12 + Git + Node 18+          (one-time)
2. npm install -g @anthropic-ai/claude-code                 (one-time)
3. claude /login                                            (one-time)
4. Ctrl+Shift+P → Git: Clone → paste repo URL
5. Bottom-left branch picker → origin/claude/jira-rag-agent-KTIvY
6. Install recommended extensions
7. Ctrl+Shift+P → Tasks: Run Build Task
8. Ctrl+Shift+P → Python: Select Interpreter → .venv
9. F5 → "Streamlit UI (recommended)"
10. Fill sidebar → Save to disk
11. Chat tab → "Analyze HT-1234"
```

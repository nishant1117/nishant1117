# Jira RAG Agent

A browser-based tool that analyzes and compares Jira epics using Claude. Runs
locally in VS Code. Uses your **Claude.ai subscription** (no API key needed) via
Claude Code.

Three ways to use it:
1. **Chat** — type a natural-language prompt, Claude picks the right tool.
2. **Analyze Epic** — form-driven, with checkboxes for extra context.
3. **Compare Epics** — two epic keys, Claude produces a delta analysis.

---

## Install once (one-time setup)

Install these on your computer if you don't already have them:

| Tool | Download | Note |
|---|---|---|
| **VS Code** | <https://code.visualstudio.com/> | |
| **Python 3.11 or 3.12** | <https://www.python.org/downloads/> | **Windows:** tick ✅ *"Add python.exe to PATH"* |
| **Git** | <https://git-scm.com/downloads> | |
| **Node.js 18+** | <https://nodejs.org/> | Needed for Claude Code |

Then open a terminal and run these two commands once:

```bash
npm install -g @anthropic-ai/claude-code
claude /login
```

The second command opens your browser. Sign in with your **Claude.ai** account
(the same one you chat with at claude.ai). From now on, every Claude call this
app makes counts against your Claude.ai subscription — no separate API bill.

> **Plan matters:** Claude Pro has tighter weekly limits than Max. A large
> epic can burn through Pro's weekly cap in one analysis. Max is strongly
> recommended.

---

## Get the code into VS Code

1. Open VS Code.
2. Press **Ctrl + Shift + P** (Mac: **Cmd + Shift + P**).
3. Type `Git: Clone` → Enter.
4. Paste this URL → Enter:
   ```
   https://github.com/nishant1117/nishant1117.git
   ```
5. Pick a folder on your disk. Click **Open** when the "Open cloned repository?" toast appears.
6. **Switch to the feature branch.** Click the word `main` in the bottom-left status bar → dropdown appears → pick **`origin/claude/jira-rag-agent-KTIvY`**.
7. Click **Install** on the "Install recommended extensions" toast.

---

## Install project dependencies

1. Press **Ctrl + Shift + P** → `Tasks: Run Build Task` → Enter.
2. Wait 1–3 minutes for pip to finish.
3. Press **Ctrl + Shift + P** → `Python: Select Interpreter` → pick the one ending in **`.venv`**.

---

## Run the app

1. Press **F5**.
2. Pick **"Streamlit UI (recommended)"** from the dropdown.
3. A browser tab opens at <http://localhost:8501>.

---

## First-time credential setup

In the left sidebar of the app:

| Field | Value |
|---|---|
| **Claude authentication** | pick *"Claude.ai subscription (Claude Code)"* |
| **Jira URL** | `https://healthtap.atlassian.net` |
| **Jira email** | `nishant.mahajan@healthtap.com` |
| **Jira API token** | your token from <https://id.atlassian.com/manage-profile/security/api-tokens> |
| **Claude model** | leave as `claude-sonnet-4-5` |

Click **Save to disk**. The credentials are saved to a local file
(`.streamlit/secrets.toml`) that is gitignored — it never leaves your
computer. Next time you run the app, the fields auto-fill.

---

## Using the tool

### 💬 Chat tab
Type a natural-language request. Claude decides whether to analyze one epic
or compare two. Examples:

- `Analyze epic HT-1234 and list the top regression risks`
- `What critical bugs did you find in HT-1234?`
- `Compare HT-1234 and HT-1300 on scope overlap`
- `Analyze HT-1234 including its attachments`

### 🔍 Analyze Epic tab

Form with:
- **Jira epic key** — e.g. `HT-1234`
- **Focus** — summary, test_cases, edge_cases, regression, bug_analysis, insights, recommendations, action_plan, or full
- **Optional custom question** — e.g. *"Which tickets carry the highest data-integrity risk?"*

**➕ Add extra context to the analysis** — expandable section with checkboxes:

| Checkbox | What it adds |
|---|---|
| 👶 Child tickets (subtasks) | Each ticket's subtasks |
| 🔗 Linked tickets | Linked issues (blocks, is-blocked-by, relates-to…) |
| 💬 Comments | All comments on each ticket |
| 📜 Changelog / history | Field-level change history |
| 📎 File attachments | Metadata for all files, inlined text for text-like files < 50 KB |

Toggle any combination and click **Analyze**. Changing the checkboxes
triggers a fresh Claude pass but reuses the Jira fetch, so re-running with
different toggles is fast.

### ⚖️ Compare Epics tab
Two epic keys, optional comparison goal, same context checkboxes (collapsed
by default), click **Compare**.

---

## Stop / restart

- **Stop**: click the red ⏹ square in the VS Code debug toolbar, or press **Shift + F5**.
- **Restart**: **F5** → **"Streamlit UI (recommended)"** again.
- **Hot reload**: Streamlit auto-reloads when you save any `.py` file.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `claude: command not found` | Claude Code CLI isn't installed. Run `npm install -g @anthropic-ai/claude-code`. |
| *"Please login first"* from the SDK | Run `claude /login` in a terminal. |
| *"Could not initialise the agent"* | Jira credentials are wrong. Check URL, email, and token in the sidebar and click **Use for session**. |
| Jira 401 / 403 | Invalid token or no access to the epic. Test at `https://healthtap.atlassian.net/rest/api/3/myself` in your browser. |
| `ModuleNotFoundError` | You skipped the interpreter step. **Ctrl+Shift+P** → `Python: Select Interpreter` → pick `.venv`. |
| `pip install chromadb` fails on Windows | Install **Microsoft C++ Build Tools** or use WSL2. The app still runs without ChromaDB. |
| First run takes forever | Expected — per-ticket Claude calls. Second run on the same epic is cached. |
| *"Weekly usage limit reached"* | You're on Claude Pro and hit the cap. Upgrade to Max, or switch to API-key mode in the sidebar. |
| Browser tab never opens | Look for `http://localhost:8501` in the VS Code terminal and click it. |

---

## Switching to API-key mode (optional)

If you prefer to use an Anthropic API key instead of your subscription:

1. In the sidebar, switch **Claude authentication** to *"Anthropic API key (pay-per-token)"*.
2. Paste your key into the **Anthropic API key** field that appears.
3. Click **Use for session** or **Save to disk**.

---

## Golden path cheat sheet

```
1. Install VS Code + Python 3.12 + Git + Node 18+
2. npm install -g @anthropic-ai/claude-code
3. claude /login
4. Ctrl+Shift+P → Git: Clone → https://github.com/nishant1117/nishant1117.git
5. Bottom-left branch picker → origin/claude/jira-rag-agent-KTIvY
6. Install recommended extensions
7. Ctrl+Shift+P → Tasks: Run Build Task
8. Ctrl+Shift+P → Python: Select Interpreter → .venv
9. F5 → "Streamlit UI (recommended)"
10. Sidebar → fill Jira token → Save to disk
11. Chat tab → "Analyze HT-1234"
```

---

## File reference

| File | What it does |
|---|---|
| `streamlit_app.py` | The web UI. Run this. |
| `agent_cli.py` | Terminal version of the same tool-use loop. |
| `llm_client.py` | Claude backend abstraction (API key or subscription). |
| `rag_agent.py` | Orchestrator: Jira → vector store → Claude. Holds `ContextOptions`. |
| `jira_client.py` | Jira API wrapper (tickets, subtasks, links, comments, changelog, attachments). |
| `test_generator.py` | All Claude prompts. |
| `vector_store.py` | ChromaDB retrieval (optional, degrades gracefully). |
| `config.py` | Lazy env-var reader. |
| `api.py` | Alternative FastAPI server. Not needed for the UI. |
| `.streamlit/secrets.toml` | **Local only, gitignored.** Your saved credentials. |

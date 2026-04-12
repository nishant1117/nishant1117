"""
Streamlit UI for the Jira RAG Agent.

Run:
    streamlit run streamlit_app.py

Features:
  - Sidebar form for Jira + Anthropic credentials (optional local save to
    .streamlit/secrets.toml, which is gitignored).
  - "Chat" tab: natural-language input, Claude picks analyzer vs
    comparator via tool-use and replies grounded in the tool output.
  - "Analyze Epic" tab: form-driven analyzer with a focus dropdown.
  - "Compare Epics" tab: form-driven comparator with an optional goal.

Nothing in this file or the repo contains real credentials. They live in
browser session state (and optionally in .streamlit/secrets.toml which
is gitignored).
"""
from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


def _extract_epic_key(input_str: str) -> str:
    """Extract epic key from various formats.
    
    Handles:
    - Just the key: "PROD-3300"
    - Jira URL: "https://healthtap.atlassian.net/browse/PROD-3300"
    - URL with query params
    """
    if not input_str:
        return ""
    
    input_str = input_str.strip()
    
    # Try to extract from URL (look for /browse/KEY pattern)
    match = re.search(r'/browse/([A-Z]+-\d+)', input_str)
    if match:
        return match.group(1)
    
    # If it's just a key like "PROD-3300", return as-is
    if re.match(r'^[A-Z]+-\d+$', input_str):
        return input_str
    
    # Otherwise return the cleaned input (might be malformed, but let Jira error handle it)
    return input_str

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Jira RAG Agent",
    page_icon="🧠",
    layout="wide",
)

SECRETS_PATH = Path(".streamlit") / "secrets.toml"


# ---------------------------------------------------------------------------
# Load saved credentials (if any) from .streamlit/secrets.toml
# ---------------------------------------------------------------------------
def _load_saved_secrets() -> Dict[str, str]:
    defaults = {
        "jira_host": os.getenv("JIRA_HOST", "https://healthtap.atlassian.net"),
        "jira_email": os.getenv("JIRA_EMAIL", ""),
        "jira_token": os.getenv("JIRA_API_TOKEN", ""),
        "claude_oauth_token": os.getenv("CLAUDE_CODE_OAUTH_TOKEN", ""),
        "model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
    }
    try:
        # st.secrets is dict-like; reading a missing key raises.
        sec = st.secrets
        for k in defaults:
            if k in sec:
                defaults[k] = str(sec[k])
    except Exception:
        pass
    return defaults


def _save_secrets_to_disk(values: Dict[str, Any]) -> None:
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Saved locally by the Streamlit UI. Gitignored — never commit.",
    ]
    for k, v in values.items():
        if v:  # Only save non-empty values
            escaped = str(v).replace('"', '\\"')
            lines.append(f'{k} = "{escaped}"')
    SECRETS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _init_session_state() -> None:
    if "_creds_loaded" in st.session_state:
        return
    defaults = _load_saved_secrets()
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.session_state._creds_loaded = True


def _claude_agent_sdk_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except Exception:
        return False


_init_session_state()


# ---------------------------------------------------------------------------
# Auto-pick subscription mode when the Claude Agent SDK is importable.
#
# The old radio widget was sticky — once a user clicked "Anthropic API key"
# its state survived reruns even if session_state.auth_mode was reset. To
# kill that class of bug we:
#   1. Use a versioned widget key (_radio_vN). Bumping N forcibly discards
#      any old widget state left over from a previous version of this file.
#   2. If claude-agent-sdk is importable, we hard-seed the widget to
#      subscription mode so it visually matches the backend that will
#      actually run.
# ---------------------------------------------------------------------------
_RADIO_KEY = "auth_mode_radio_v3"
_LABEL_SUB = "Claude.ai subscription (Claude Code)"
_LABEL_API = "Anthropic API key (pay-per-token)"

if _RADIO_KEY not in st.session_state:
    if _claude_agent_sdk_available():
        st.session_state[_RADIO_KEY] = _LABEL_SUB
        st.session_state.auth_mode = "subscription"
    else:
        st.session_state[_RADIO_KEY] = _LABEL_API
        st.session_state.auth_mode = "api"


# ---------------------------------------------------------------------------
# Sidebar: credentials form
# ---------------------------------------------------------------------------
st.sidebar.title("Credentials")
st.sidebar.markdown(
    "Enter once, then use the tabs on the right. These values live in "
    "this browser session only, unless you click **Save to disk** "
    "(which writes to `.streamlit/secrets.toml`, a gitignored file on "
    "your computer)."
)

st.session_state.jira_host = st.sidebar.text_input(
    "Jira URL",
    value=st.session_state.jira_host,
    help="e.g. https://your-domain.atlassian.net",
)
st.session_state.jira_email = st.sidebar.text_input(
    "Jira email",
    value=st.session_state.jira_email,
)
st.session_state.jira_token = st.sidebar.text_input(
    "Jira API token",
    value=st.session_state.jira_token,
    type="password",
    help="Generate at https://id.atlassian.com/manage-profile/security/api-tokens",
)
st.session_state.claude_oauth_token = st.sidebar.text_input(
    "Claude OAuth Token",
    value=st.session_state.claude_oauth_token,
    type="password",
    help="Run 'claude setup-token' to generate a long-lived token",
)
st.session_state.model = st.sidebar.text_input(
    "Claude model",
    value=st.session_state.model,
)

col_a, col_b = st.sidebar.columns(2)
if col_a.button("Use for session", use_container_width=True):
    st.sidebar.success("Credentials active for this session.")
if col_b.button("Save to disk", use_container_width=True):
    try:
        _save_secrets_to_disk({
            "jira_host": st.session_state.jira_host,
            "jira_email": st.session_state.jira_email,
            "jira_token": st.session_state.jira_token,
            "claude_oauth_token": st.session_state.claude_oauth_token,
            "model": st.session_state.model,
        })
        st.sidebar.success(
            f"Saved to `{SECRETS_PATH}`. The file is gitignored."
        )
    except Exception as e:
        st.sidebar.error(f"Could not save: {e}")

# Push credentials into the environment so config.py (lazy) picks them up.
os.environ["JIRA_HOST"] = st.session_state.jira_host or ""
os.environ["JIRA_EMAIL"] = st.session_state.jira_email or ""
os.environ["JIRA_API_TOKEN"] = st.session_state.jira_token or ""
os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = st.session_state.claude_oauth_token or ""
os.environ["CLAUDE_MODEL"] = st.session_state.model or "claude-sonnet-4-6"


def _credentials_ok() -> bool:
    return all([
        st.session_state.jira_host,
        st.session_state.jira_email,
        st.session_state.jira_token,
        st.session_state.claude_oauth_token,
    ])


# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("🧠 Jira RAG Agent")

# Prominent mode banner so the user always knows what they're using.
if st.session_state.auth_mode == "subscription":
    st.success(
        "**Auth mode: Claude.ai subscription (Claude Code)** — no API key "
        "needed. Calls count against your Claude.ai plan."
    )
    if not _claude_agent_sdk_available():
        st.error(
            "⚠️ `claude-agent-sdk` is not installed in this Python venv, "
            "so subscription mode cannot make LLM calls. Fix it with one "
            "of these:\n\n"
            "1. Open a VS Code terminal (Terminal → New Terminal) and run:\n"
            "   ```\n"
            "   pip install claude-agent-sdk\n"
            "   ```\n"
            "2. Then make sure Claude Code itself is installed and you're "
            "logged in:\n"
            "   ```\n"
            "   npm install -g @anthropic-ai/claude-code\n"
            "   claude /login\n"
            "   ```\n"
            "3. Restart the Streamlit app (Shift + F5, then F5 again).\n\n"
            "Alternatively, flip the **Claude authentication** radio in "
            "the sidebar to *'Anthropic API key'* and paste a key."
        )
        st.stop()
else:
    st.info(
        "**Auth mode: Anthropic API key** — each call is billed per "
        "token against the key below. Switch to subscription mode in "
        "the sidebar if you'd rather use your Claude.ai plan."
    )

st.caption(
    "Analyze and compare Jira epics with Claude. Uses a tool-use loop so "
    "Claude picks the right tool (analyzer / comparator) for your prompt."
)

if not _credentials_ok():
    st.info(
        "Fill in the credentials in the left sidebar and click **Use for "
        "session** or **Save to disk**. Then come back here."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Lazy imports — only after credentials are set so the Jira / Anthropic
# clients initialise against the right values.
# ---------------------------------------------------------------------------
try:
    from agent_cli import AgentSession, run_turn
    from llm_client import get_backend
    from rag_agent import ContextOptions
except Exception as e:  # pragma: no cover
    st.error(f"Failed to import agent modules: {e}")
    st.code(traceback.format_exc())
    st.stop()


@st.cache_resource(show_spinner=False)
def get_session_bundle(jira_host: str, jira_email: str, claude_oauth_token: str):
    """Bundle of long-lived state.

    Caching on (jira_host, jira_email, claude_oauth_token) means
    the session is rebuilt when credentials change but reused across
    reruns otherwise.
    """
    session = AgentSession()
    return {"session": session, "history": []}


try:
    bundle = get_session_bundle(
        st.session_state.get("jira_host", ""),
        st.session_state.get("jira_email", ""),
        st.session_state.get("claude_oauth_token", ""),
    )
except Exception as e:
    st.error(f"Could not initialise the agent: {e}")
    st.code(traceback.format_exc())
    st.stop()


# ---------------------------------------------------------------------------
# Helpers for pretty display of tool results
# ---------------------------------------------------------------------------
def _render_test_cases(tcs: List[Dict[str, Any]]) -> None:
    if not tcs:
        st.caption("No test cases.")
        return
    for tc in tcs:
        with st.expander(f"{tc.get('id', '')} — {tc.get('title', '')}"):
            st.write(f"**Priority:** {tc.get('priority', 'N/A')}")
            st.write(f"**Category:** {tc.get('category', 'N/A')}")
            if tc.get("description"):
                st.write(f"**Description:** {tc['description']}")
            if tc.get("preconditions"):
                st.write("**Preconditions:**")
                for p in tc["preconditions"]:
                    st.write(f"- {p}")
            if tc.get("steps"):
                st.write("**Steps:**")
                for s in tc["steps"]:
                    st.write(f"- {s}")
            st.write(f"**Expected result:** {tc.get('expected_result', 'N/A')}")


def _render_edge_cases(ecs: List[Dict[str, Any]]) -> None:
    if not ecs:
        st.caption("No edge cases.")
        return
    for ec in ecs:
        with st.expander(f"{ec.get('id', '')} — {ec.get('title', '')}"):
            st.write(f"**Severity:** {ec.get('severity', 'N/A')}")
            st.write(f"**Category:** {ec.get('category', 'N/A')}")
            st.write(f"**Trigger:** {ec.get('trigger_condition', 'N/A')}")
            st.write(f"**Expected behavior:** {ec.get('expected_behavior', 'N/A')}")


def _render_generic(data: Any) -> None:
    st.json(data, expanded=False)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chat, tab_analyze, tab_compare, tab_attach = st.tabs(
    ["💬 Chat", "🔍 Analyze Epic", "⚖️ Compare Epics", "📎 Attachment Analyzer"]
)

# --- Chat tab --------------------------------------------------------------
with tab_chat:
    st.subheader("Ask the agent anything")
    st.caption(
        "Claude will decide whether to run the analyzer or comparator "
        "based on your prompt. Examples: "
        "*'Analyze HT-1234 and show the critical bugs'*, "
        "*'Compare HT-1234 and HT-1300 on regression risk'*."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Your request..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            # Create placeholder and progress tracking
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            response_placeholder = st.empty()
            
            try:
                # Create a progress tracker callback
                progress_state = {"status": "", "progress": 0.0, "details": ""}
                
                # Use container for real-time updates
                with st.container():
                    reply = run_turn(
                        bundle["session"],
                        bundle["history"],
                        prompt,
                        progress_callback=lambda status, progress, details: (
                            progress_state.update({"status": status, "progress": progress, "details": details}),
                            status_placeholder.write(f"📊 {status}"),
                            progress_placeholder.progress(min(progress, 1.0), text=details if details else None),
                        )
                    )
                # Clear progress indicators once done
                status_placeholder.empty()
                progress_placeholder.empty()
            except Exception as e:
                reply = f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```"
                status_placeholder.empty()
                progress_placeholder.empty()
            response_placeholder.markdown(reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    if st.button("Clear chat"):
        st.session_state.chat_messages = []
        bundle["history"] = []
        st.rerun()

# --- Analyze tab -----------------------------------------------------------
with tab_analyze:
    st.subheader("Run the analyzer on a single epic")

    epic_key = st.text_input("Jira epic key", placeholder="e.g. HT-1234")
    focus = st.selectbox(
        "What do you want to see?",
        [
            "summary",
            "test_cases",
            "edge_cases",
            "regression",
            "bug_analysis",
            "insights",
            "recommendations",
            "action_plan",
            "full",
        ],
        index=0,
        help=(
            "Pick the narrowest slice that answers your question. "
            "'full' returns everything but is verbose."
        ),
    )
    custom_q = st.text_input(
        "Optional custom question",
        placeholder="e.g. Which tickets carry the highest data-integrity risk?",
    )

    with st.expander("➕ Add extra context to the analysis", expanded=True):
        st.caption(
            "Toggle which extra Jira data to feed into Claude alongside the "
            "ticket fields. More context = more grounded answers, but "
            "slower and uses more of your Claude quota. Changing these "
            "options re-runs the analysis."
        )
        col_l, col_r = st.columns(2)
        include_subtasks = col_l.checkbox(
            "👶 Child tickets (subtasks)",
            value=True,
            help="Pull in each ticket's subtasks with key, status, and summary.",
        )
        include_linked = col_l.checkbox(
            "🔗 Linked tickets",
            value=True,
            help="Include linked issues (blocks, is-blocked-by, relates-to, etc).",
        )
        include_comments = col_l.checkbox(
            "💬 Comments",
            value=True,
            help="Include every comment on each ticket (author + body).",
        )
        include_changelog = col_r.checkbox(
            "📜 Changelog / history",
            value=True,
            help="Include the change history (field + from/to values).",
        )
        include_attachments = col_r.checkbox(
            "📎 File attachments",
            value=False,
            help=(
                "Include attachment metadata for every attachment, and inline "
                "the text content of attachments that look like text files "
                "(< 50 KB). Binary files are referenced by filename only."
            ),
        )

    context_options = ContextOptions(
        include_subtasks=include_subtasks,
        include_linked=include_linked,
        include_comments=include_comments,
        include_changelog=include_changelog,
        include_attachments=include_attachments,
    )

    if st.button("Analyze", type="primary", disabled=not epic_key):
        clean_key = _extract_epic_key(epic_key)
        status_box = st.status(
            f"Starting analysis of {clean_key}...", expanded=True
        )
        progress_bar = st.progress(0.0)
        step_log = st.empty()

        def _on_progress(msg: str, pct: float) -> None:
            progress_bar.progress(min(pct, 1.0))
            step_log.caption(f"⏳ {msg}")
            status_box.update(label=msg)

        try:
            result = bundle["session"].run_analyzer(
                clean_key,
                focus,
                custom_q.strip() or None,
                context_options=context_options,
                progress=_on_progress,
            )
            status_box.update(
                label=f"Analysis of {clean_key} complete!", state="complete"
            )
        except Exception as e:
            status_box.update(label=f"Analysis failed: {e}", state="error")
            st.code(traceback.format_exc())
            result = None
        finally:
            progress_bar.empty()
            step_log.empty()

        if result:
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Done. Showing `{focus}` for {epic_key}.")

                if focus == "summary":
                    st.metric("Tickets", result.get("ticket_count", 0))
                    st.write(f"**Summary:** {result.get('summary', '')}")
                    st.write(f"**Status:** {result.get('status', '')}")
                    st.write(f"**Priority:** {result.get('priority', '')}")
                    st.write("**Ticket keys:**")
                    st.write(", ".join(result.get("ticket_keys", [])))
                elif focus == "test_cases":
                    for tk, tcs in result.get("test_cases_by_ticket", {}).items():
                        st.markdown(f"### {tk}")
                        _render_test_cases(tcs)
                elif focus == "edge_cases":
                    for tk, ecs in result.get("edge_cases_by_ticket", {}).items():
                        st.markdown(f"### {tk}")
                        _render_edge_cases(ecs)
                else:
                    _render_generic(result)

                if result.get("custom_answer"):
                    st.markdown("### Custom question answer")
                    _render_generic(result["custom_answer"])

# --- Compare tab -----------------------------------------------------------
with tab_compare:
    st.subheader("Compare two epics")

    c1, c2 = st.columns(2)
    epic_1 = c1.text_input("First epic key", placeholder="e.g. HT-1234")
    epic_2 = c2.text_input("Second epic key", placeholder="e.g. HT-1300")
    goal = st.text_input(
        "Comparison goal (optional)",
        placeholder="e.g. Which has higher regression risk?",
    )

    with st.expander("➕ Add extra context to the comparison", expanded=False):
        col_l2, col_r2 = st.columns(2)
        cmp_subtasks = col_l2.checkbox(
            "👶 Child tickets", value=True, key="cmp_subtasks"
        )
        cmp_linked = col_l2.checkbox(
            "🔗 Linked tickets", value=True, key="cmp_linked"
        )
        cmp_comments = col_l2.checkbox(
            "💬 Comments", value=True, key="cmp_comments"
        )
        cmp_changelog = col_r2.checkbox(
            "📜 Changelog", value=True, key="cmp_changelog"
        )
        cmp_attach = col_r2.checkbox(
            "📎 Attachments", value=False, key="cmp_attach"
        )
    compare_context = ContextOptions(
        include_subtasks=cmp_subtasks,
        include_linked=cmp_linked,
        include_comments=cmp_comments,
        include_changelog=cmp_changelog,
        include_attachments=cmp_attach,
    )

    if st.button(
        "Compare", type="primary", disabled=not (epic_1 and epic_2)
    ):
        clean_key_1 = _extract_epic_key(epic_1)
        clean_key_2 = _extract_epic_key(epic_2)
        cmp_status = st.status(
            f"Comparing {clean_key_1} vs {clean_key_2}...", expanded=True
        )
        cmp_bar = st.progress(0.0)
        cmp_log = st.empty()

        def _on_cmp(msg: str, pct: float) -> None:
            cmp_bar.progress(min(pct, 1.0))
            cmp_log.caption(f"⏳ {msg}")
            cmp_status.update(label=msg)

        try:
            result = bundle["session"].run_comparator(
                clean_key_1,
                clean_key_2,
                goal.strip() or None,
                context_options=compare_context,
                progress=_on_cmp,
            )
            cmp_status.update(label="Comparison complete!", state="complete")
        except Exception as e:
            cmp_status.update(label=f"Comparison failed: {e}", state="error")
            st.code(traceback.format_exc())
            result = None
        finally:
            cmp_bar.empty()
            cmp_log.empty()

        if result:
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Compared {clean_key_1} vs {clean_key_2}.")
                col_l, col_r = st.columns(2)
                col_l.markdown(f"### {clean_key_1}")
                col_l.json(result.get("epic_1_summary", {}))
                col_r.markdown(f"### {clean_key_2}")
                col_r.json(result.get("epic_2_summary", {}))

                st.markdown("### Delta analysis")
                _render_generic(result.get("comparison", {}))

# --- Attachment Analyzer tab -----------------------------------------------
with tab_attach:
    st.subheader("Upload & analyze files with Claude")
    st.markdown(
        "Drag and drop any files and Claude will analyze them. "
        "Supported formats:\n\n"
        "| Type | How it's parsed |\n"
        "|---|---|\n"
        "| **Images** (PNG, JPG, GIF, WebP) | Claude **sees** them via vision |\n"
        "| **PDFs** | Text extracted page-by-page via `pypdf` |\n"
        "| **Excel** (.xlsx, .xls) | All sheets parsed — columns, data types, stats, rows |\n"
        "| **CSV / TSV** | Parsed with pandas — columns, stats, full data |\n"
        "| **Text / Code** (.txt, .md, .json, .yaml, .py, .sql, etc.) | Read as UTF-8 |\n"
        "| **Other binary** | Listed by filename only |"
    )

    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=None,  # accept all file types
        help=(
            "Upload multiple files at once. Excel and CSV files are fully "
            "parsed (all sheets, all rows up to 5000, column types, summary "
            "stats). PDFs are text-extracted. Images are sent to Claude's "
            "vision so it can see screenshots and diagrams."
        ),
    )

    attach_epic_key = st.text_input(
        "Optional: Jira epic key for additional context",
        placeholder="e.g. PROD-3761 — leave blank to analyze files standalone",
        key="attach_epic_key",
    )

    attach_question = st.text_area(
        "What would you like to know about these files?",
        placeholder=(
            "Examples:\n"
            "• Analyze these screenshots and identify UI issues\n"
            "• What does this spec document say about the acceptance criteria?\n"
            "• Compare these two mockups and list the differences\n"
            "• Extract all test scenarios from this PDF\n"
            "• What bugs or issues do you see in these screenshots?"
        ),
        height=120,
        key="attach_question",
    )

    if uploaded_files:
        with st.expander(
            f"📁 {len(uploaded_files)} file(s) selected", expanded=True
        ):
            for uf in uploaded_files:
                size_kb = len(uf.getvalue()) / 1024
                lower = (uf.name or "").lower()
                if uf.type and uf.type.startswith("image/"):
                    icon = "🖼️"
                    how = "vision"
                elif lower.endswith(".pdf") or uf.type == "application/pdf":
                    icon = "📄"
                    how = "PDF text extraction"
                elif lower.endswith((".xlsx", ".xls")):
                    icon = "📊"
                    how = "Excel parser (all sheets)"
                elif lower.endswith((".csv", ".tsv")):
                    icon = "📊"
                    how = "CSV/TSV parser (pandas)"
                else:
                    icon = "📝"
                    how = "text"
                st.caption(
                    f"{icon} **{uf.name}** — {size_kb:.1f} KB → *{how}*"
                )

    can_analyze = bool(uploaded_files) and bool(attach_question and attach_question.strip())

    if st.button(
        "Analyze files", type="primary", disabled=not can_analyze,
        key="analyze_files_btn",
    ):
        from llm_client import get_backend

        backend = get_backend()
        attach_status = st.status("Preparing files for Claude...", expanded=True)
        attach_bar = st.progress(0.0)

        # Build the file list for the multimodal call.
        file_dicts: List[Dict[str, Any]] = []
        for i, uf in enumerate(uploaded_files):
            raw = uf.getvalue()
            mt = uf.type or "application/octet-stream"
            file_dicts.append({
                "name": uf.name,
                "media_type": mt,
                "data": raw,
            })
            attach_status.update(
                label=f"Reading file {i + 1}/{len(uploaded_files)}: {uf.name}"
            )
            attach_bar.progress((i + 1) / (len(uploaded_files) + 1))

        # Build the text prompt with optional Jira context.
        prompt_parts = [attach_question.strip()]
        if attach_epic_key and attach_epic_key.strip():
            clean_ek = _extract_epic_key(attach_epic_key)
            try:
                attach_status.update(
                    label=f"Fetching Jira context for {clean_ek}..."
                )
                epic_data = bundle["session"].agent.jira_client.get_epic_details(
                    clean_ek
                )
                if epic_data:
                    prompt_parts.append(
                        f"\n\nJira context for epic {clean_ek}:\n"
                        f"Summary: {epic_data.get('summary', '')}\n"
                        f"Description: {epic_data.get('description', '')}\n"
                        f"Status: {epic_data.get('status', '')}\n"
                        f"Priority: {epic_data.get('priority', '')}"
                    )
            except Exception as e:
                st.warning(f"Could not fetch Jira context: {e}")

        system = (
            "You are an expert software analyst with vision capabilities. "
            "Analyze the provided files thoroughly. For images "
            "(screenshots, mockups, diagrams), describe what you see in "
            "detail and identify issues, patterns, or insights. For text "
            "files and PDFs, extract key information. Provide structured, "
            "actionable analysis. If a Jira epic context is provided, "
            "relate your findings to that epic's scope."
        )

        attach_status.update(label="Claude is analyzing your files...")
        attach_bar.progress(0.8)

        try:
            result_text = backend.complete_with_files(
                system=system,
                text_prompt="\n".join(prompt_parts),
                files=file_dicts,
            )
            attach_status.update(
                label=f"Analysis complete — {len(uploaded_files)} file(s) processed!",
                state="complete",
            )
        except Exception as e:
            result_text = None
            attach_status.update(label=f"Analysis failed: {e}", state="error")
            st.code(traceback.format_exc())
        finally:
            attach_bar.empty()

        if result_text:
            st.markdown("### Analysis results")
            st.markdown(result_text)

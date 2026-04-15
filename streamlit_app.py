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
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# Load .env file to populate os.environ
load_dotenv()


# ---------------------------------------------------------------------------
# Claude Code credential auto-discovery
# ---------------------------------------------------------------------------
def _read_claude_credentials_file() -> Optional[str]:
    """Best-effort read of the OAuth token Claude Code stores on disk
    after `claude /login`. Returns None if not found.

    Locations (in priority order) across macOS / Linux installs:
      ~/.claude/.credentials.json
      ~/.config/claude/credentials.json
      ~/.claude/credentials.json
    """
    home = Path.home()
    candidates = [
        home / ".claude" / ".credentials.json",
        home / ".config" / "claude" / "credentials.json",
        home / ".claude" / "credentials.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Common shapes: {"access_token": "..."} or
            # {"claudeAiOauth": {"accessToken": "..."}}
            if isinstance(data, dict):
                token = (
                    data.get("access_token")
                    or data.get("accessToken")
                    or data.get("claudeAiOauth", {}).get("accessToken")
                )
                if token:
                    return str(token)
        except Exception:
            continue
    return None


def _run_claude_setup_token() -> tuple[bool, str]:
    """Attempt to launch `claude setup-token` in a subprocess and
    capture the generated OAuth token. Returns (success, message_or_token)."""
    claude = shutil.which("claude")
    if not claude:
        return False, (
            "The `claude` CLI was not found on PATH. Install it with "
            "`npm install -g @anthropic-ai/claude-code` and restart."
        )
    try:
        result = subprocess.run(
            [claude, "setup-token"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            return False, (
                f"`claude setup-token` exited with code "
                f"{result.returncode}. stderr: {result.stderr[:400]}"
            )
        # Token patterns: OAuth tokens typically start with "sk-ant-oat01-"
        # or similar. Grab the first plausible long token from stdout.
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        match = re.search(r"(sk-ant-[A-Za-z0-9_\-]{20,})", output)
        if match:
            return True, match.group(1)
        # Fall back to reading from the credentials file that setup-token
        # writes.
        token = _read_claude_credentials_file()
        if token:
            return True, token
        return False, (
            "`claude setup-token` ran but no token was captured. "
            "Open a terminal, run `claude /login`, then come back and "
            "click 'Detect saved login' below."
        )
    except subprocess.TimeoutExpired:
        return False, (
            "`claude setup-token` timed out. Run it manually in a "
            "terminal: `claude setup-token`."
        )
    except Exception as e:
        return False, f"Failed to run `claude setup-token`: {e}"


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
# Auto-detect saved Claude OAuth token on first load so we can prefill it
# ---------------------------------------------------------------------------
if not st.session_state.get("claude_oauth_token"):
    disk_token = _read_claude_credentials_file()
    if disk_token:
        st.session_state.claude_oauth_token = disk_token


def _credentials_ok() -> bool:
    jira_ok = all([
        st.session_state.get("jira_host"),
        st.session_state.get("jira_email"),
        st.session_state.get("jira_token"),
    ])
    if not jira_ok:
        return False
    return bool(st.session_state.get("claude_oauth_token"))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧠 Jira RAG Agent")

# ---------------------------------------------------------------------------
# Credential wizard — one field at a time until everything's filled.
# Once credentials are all set, this collapses into a small sidebar that
# lets the user reconfigure.
# ---------------------------------------------------------------------------
WIZARD_STEPS = [
    ("jira_host", "Jira URL",
     "e.g. https://your-domain.atlassian.net", False),
    ("jira_email", "Jira email",
     "The email associated with your Atlassian account.", False),
    ("jira_token", "Jira API token",
     "Generate at https://id.atlassian.com/manage-profile/security/api-tokens",
     True),
    ("claude_oauth_token", "Claude OAuth token",
     "Run `claude setup-token` in a terminal, or click the button below "
     "to run it for you.", True),
]


def _current_wizard_step() -> Optional[tuple]:
    for key, label, help_text, is_password in WIZARD_STEPS:
        if not st.session_state.get(key):
            return key, label, help_text, is_password
    return None


if not _credentials_ok():
    # Wizard mode — ask for missing fields one at a time.
    st.markdown("### 🔐 Let's set up your credentials")
    st.caption(
        "One step at a time. Everything stays in your browser session "
        "(or locally on disk — never pushed to git)."
    )

    # Render a progress indicator showing which step we're on.
    step_indicators = []
    for key, label, _, _ in WIZARD_STEPS:
        if st.session_state.get(key):
            step_indicators.append(f"✅ **{label}**")
        else:
            step_indicators.append(f"⬜ {label}")
    st.markdown(" → ".join(step_indicators))
    st.markdown("---")

    current = _current_wizard_step()
    if current is None:
        st.rerun()

    key, label, help_text, is_password = current
    st.subheader(f"Step: {label}")
    st.caption(help_text)

    # For the Claude token step, offer a "login" button that runs
    # `claude setup-token` and auto-fills.
    if key == "claude_oauth_token":
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button(
                "🔐 Login to Claude now",
                use_container_width=True,
                help=(
                    "Runs `claude setup-token` in the background to generate "
                    "an OAuth token. Opens a browser window to sign in."
                ),
            ):
                with st.spinner("Running `claude setup-token`..."):
                    ok, msg = _run_claude_setup_token()
                if ok:
                    st.session_state.claude_oauth_token = msg
                    st.success("Token captured. Continuing...")
                    st.rerun()
                else:
                    st.error(msg)
        with col_b:
            if st.button(
                "🔍 Detect saved login",
                use_container_width=True,
                help="Read token from ~/.claude/.credentials.json if present.",
            ):
                tok = _read_claude_credentials_file()
                if tok:
                    st.session_state.claude_oauth_token = tok
                    st.success("Saved token found. Continuing...")
                    st.rerun()
                else:
                    st.warning(
                        "No saved token found. Run `claude /login` in a "
                        "terminal first, then retry."
                    )

        st.caption(
            "Or paste a token manually (generate via `claude setup-token`):"
        )

    value = st.text_input(
        label,
        type="password" if is_password else "default",
        key=f"wizard_input_{key}",
        placeholder="Paste here and press Enter",
    )
    if value:
        st.session_state[key] = value
        st.rerun()

    st.markdown("---")
    with st.expander("Show all credentials at once"):
        st.caption(
            "Prefer a flat form? Fill everything below and submit. Values "
            "with the ✅ above are already captured."
        )
        jh = st.text_input(
            "Jira URL",
            value=st.session_state.get("jira_host", "https://healthtap.atlassian.net"),
            key="flat_jh",
        )
        je = st.text_input(
            "Jira email",
            value=st.session_state.get("jira_email", ""),
            key="flat_je",
        )
        jt = st.text_input(
            "Jira API token",
            value=st.session_state.get("jira_token", ""),
            type="password",
            key="flat_jt",
        )
        co = st.text_input(
            "Claude OAuth token",
            value=st.session_state.get("claude_oauth_token", ""),
            type="password",
            key="flat_co",
        )
        if st.button("Save all and continue"):
            if jh and je and jt and co:
                st.session_state.jira_host = jh
                st.session_state.jira_email = je
                st.session_state.jira_token = jt
                st.session_state.claude_oauth_token = co
                st.rerun()
            else:
                st.error("All fields are required.")

    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: compact view of live credentials + reconfigure button
# ---------------------------------------------------------------------------
st.sidebar.success("✅ Credentials active")
with st.sidebar.expander("Active credentials (masked)", expanded=False):
    st.caption(f"**Jira URL:** {st.session_state.jira_host}")
    st.caption(f"**Jira email:** {st.session_state.jira_email}")
    _jt = st.session_state.jira_token or ""
    _ct = st.session_state.claude_oauth_token or ""
    st.caption(
        "**Jira token:** `" + "•" * 8 + f"{_jt[-4:] if len(_jt) >= 4 else '****'}`"
    )
    st.caption(
        "**Claude token:** `" + "•" * 8
        + f"{_ct[-4:] if len(_ct) >= 4 else '****'}`"
    )
    st.caption(f"**Model:** {st.session_state.model}")

if st.sidebar.button("🔁 Reconfigure credentials", use_container_width=True):
    for k in ("jira_host", "jira_email", "jira_token", "claude_oauth_token"):
        st.session_state.pop(k, None)
    st.rerun()
if st.sidebar.button("💾 Save credentials to disk", use_container_width=True):
    try:
        _save_secrets_to_disk({
            "jira_host": st.session_state.jira_host,
            "jira_email": st.session_state.jira_email,
            "jira_token": st.session_state.jira_token,
            "claude_oauth_token": st.session_state.claude_oauth_token,
            "model": st.session_state.model,
        })
        st.sidebar.success(
            f"Saved to `{SECRETS_PATH}` (gitignored)."
        )
    except Exception as e:
        st.sidebar.error(f"Could not save: {e}")

# Push credentials into the environment so config.py (lazy) picks them up.
os.environ["JIRA_HOST"] = st.session_state.jira_host or ""
os.environ["JIRA_EMAIL"] = st.session_state.jira_email or ""
os.environ["JIRA_API_TOKEN"] = st.session_state.jira_token or ""
os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = st.session_state.claude_oauth_token or ""
os.environ["CLAUDE_MODEL"] = st.session_state.model or "claude-sonnet-4-6"

# Banner showing auth mode + extended thinking state.
thinking_on = os.getenv("CLAUDE_EXTENDED_THINKING", "true").lower() in (
    "1", "true", "yes", "on"
)
if st.session_state.auth_mode == "subscription":
    st.success(
        "**Auth mode: Claude.ai subscription (Claude Code)** · "
        f"Model: `{st.session_state.model}` · "
        f"Extended thinking: {'🧠 enabled' if thinking_on else 'off'}"
    )
    if not _claude_agent_sdk_available():
        st.error(
            "⚠️ `claude-agent-sdk` is not installed in this Python venv.\n\n"
            "Fix: open the VS Code terminal and run "
            "`pip install claude-agent-sdk`, then restart the app."
        )
        st.stop()
else:
    st.info(
        "**Auth mode: Anthropic API key** · "
        f"Model: `{st.session_state.model}` · "
        f"Extended thinking: {'🧠 enabled' if thinking_on else 'off'}"
    )

st.caption(
    "Analyze and compare Jira epics with Claude. Uses a tool-use loop so "
    "Claude picks the right tool (analyzer / comparator / jira_search) "
    "for your prompt."
)


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
        "Claude will decide whether to run the analyzer, comparator, or "
        "JQL search based on your prompt. Upload files alongside your "
        "question to include them in the analysis."
    )

    chat_files = st.file_uploader(
        "📎 Attach files to your next message (optional)",
        accept_multiple_files=True,
        type=None,
        key="chat_files",
        help="Uploaded files will be analyzed alongside the next prompt you send.",
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
            # Expanded status box that accumulates every step as a
            # persistent line, so the user can audit exactly what happened.
            chat_status = st.status("Thinking...", expanded=True)
            chat_bar = chat_status.progress(0.0)
            st.session_state.chat_progress_log = []
            chat_log_container = chat_status.empty()
            response_placeholder = st.empty()
            status_placeholder = st.empty()
            progress_placeholder = st.empty()

            def _chat_progress(status: str, progress: float, details: str = "") -> None:
                line = f"⏳ {status}"
                if details and details != status:
                    line += f" — {details}"
                st.session_state.chat_progress_log.append(line)
                chat_log_container.markdown(
                    "\n".join(
                        f"- {l}" for l in st.session_state.chat_progress_log
                    )
                )
                chat_bar.progress(min(progress, 1.0))
                chat_status.update(label=status)

            try:
                reply = run_turn(
                    bundle["session"],
                    bundle["history"],
                    prompt,
                    progress_callback=_chat_progress,
                )
                # If files were uploaded, do a supplementary analysis
                # that combines the tool result with the file contents.
                if chat_files and reply and not reply.startswith("**Error"):
                    status_placeholder.write("📎 Analyzing uploaded files...")
                    try:
                        from llm_client import get_backend as _get_be3
                        _be3 = _get_be3()
                        chat_file_dicts = [
                            {
                                "name": uf.name,
                                "media_type": uf.type or "application/octet-stream",
                                "data": uf.getvalue(),
                            }
                            for uf in chat_files
                        ]
                        file_reply = _be3.complete_with_files(
                            system=(
                                "You previously answered a question about "
                                "Jira data. The user also uploaded files. "
                                "Analyze the files and integrate your "
                                "findings with the previous answer. For "
                                "images describe what you see. For data "
                                "files analyze the content."
                            ),
                            text_prompt=(
                                f"Previous answer:\n{reply[:20_000]}\n\n"
                                f"Now analyze these {len(chat_file_dicts)} "
                                f"uploaded file(s) and integrate:"
                            ),
                            files=chat_file_dicts,
                        )
                        reply = (
                            reply
                            + "\n\n---\n\n### 📎 Uploaded file analysis\n\n"
                            + file_reply
                        )
                    except Exception as e:
                        reply += f"\n\n*(File analysis failed: {e})*"

                # Finalise the status box but KEEP IT EXPANDED so the
                # full step history stays visible to the user.
                chat_status.update(
                    label=f"✅ Response ready — "
                          f"{len(st.session_state.chat_progress_log)} steps",
                    state="complete",
                    expanded=True,
                )
                status_placeholder.empty()
                progress_placeholder.empty()
            except Exception as e:
                reply = f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```"
                chat_status.update(
                    label=f"❌ Error: {e}", state="error", expanded=True
                )
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

    # --- Optional file uploads to include in analysis ---
    analyze_files = st.file_uploader(
        "📎 Upload additional files to include in analysis (optional)",
        accept_multiple_files=True,
        type=None,
        key="analyze_files",
        help=(
            "Upload screenshots, PDFs, Excel sheets, CSVs, specs, or "
            "any files. They'll be analyzed together with the Jira data. "
            "Images are read via Claude vision. Excel/CSV are fully parsed."
        ),
    )
    if analyze_files:
        st.caption(
            f"{len(analyze_files)} file(s): "
            + ", ".join(f.name for f in analyze_files)
        )

    if st.button("Analyze", type="primary", disabled=not epic_key):
        clean_key = _extract_epic_key(epic_key)
        # expanded=True and we never collapse it — the user asked for all
        # progress steps to stay visible even after completion.
        status_box = st.status(
            f"Starting analysis of {clean_key}...", expanded=True
        )
        progress_bar = status_box.progress(0.0)

        # Use a running list so every step persists as a line inside the
        # status box (instead of being overwritten by the next message).
        if "analyze_progress_log" not in st.session_state:
            st.session_state.analyze_progress_log = []
        st.session_state.analyze_progress_log = []
        log_container = status_box.empty()

        def _on_progress(msg: str, pct: float) -> None:
            st.session_state.analyze_progress_log.append(f"⏳ {msg}")
            # Render the full running log each time so nothing disappears.
            log_container.markdown(
                "\n".join(
                    f"- {line}" for line in st.session_state.analyze_progress_log
                )
            )
            progress_bar.progress(min(pct, 1.0))
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
                label=f"✅ Analysis of {clean_key} complete — "
                      f"{len(st.session_state.analyze_progress_log)} steps",
                state="complete",
                expanded=True,
            )
        except Exception as e:
            status_box.update(
                label=f"❌ Analysis failed: {e}", state="error", expanded=True
            )
            st.code(traceback.format_exc())
            result = None

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

                # --- Integrated file analysis (if files uploaded) ---
                if analyze_files:
                    st.markdown("---")
                    st.markdown("### 📎 Uploaded file analysis")
                    file_status = st.status(
                        "Analyzing uploaded files with Jira context...",
                        expanded=True,
                    )
                    try:
                        from llm_client import get_backend as _get_be
                        _be = _get_be()
                        file_dicts = [
                            {
                                "name": uf.name,
                                "media_type": uf.type or "application/octet-stream",
                                "data": uf.getvalue(),
                            }
                            for uf in analyze_files
                        ]
                        # Build a summary of the Jira analysis to give
                        # Claude context alongside the files.
                        summary = json.dumps(result, default=str)[:30_000]
                        file_prompt = (
                            f"The user analyzed Jira epic {clean_key}. "
                            f"Here is a summary of the analysis:\n\n"
                            f"{summary}\n\n"
                            f"The user also uploaded {len(file_dicts)} "
                            f"file(s) for additional context. Analyze "
                            f"these files and integrate your findings with "
                            f"the Jira analysis above. Identify "
                            f"connections, gaps, issues, or insights that "
                            f"emerge from combining the Jira data with "
                            f"the uploaded files."
                        )
                        if custom_q and custom_q.strip():
                            file_prompt += (
                                f"\n\nAlso address the user's question: "
                                f"{custom_q.strip()}"
                            )
                        file_result = _be.complete_with_files(
                            system=(
                                "You are an expert analyst. You have both "
                                "Jira epic analysis results and uploaded "
                                "files. Provide an integrated analysis "
                                "combining insights from both sources. "
                                "For images, describe what you see. For "
                                "data files, analyze the data. Relate "
                                "everything to the Jira epic context."
                            ),
                            text_prompt=file_prompt,
                            files=file_dicts,
                        )
                        file_status.update(
                            label="File analysis complete!", state="complete"
                        )
                        st.markdown(file_result)
                    except Exception as e:
                        file_status.update(
                            label=f"File analysis failed: {e}", state="error"
                        )
                        st.code(traceback.format_exc())

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

    compare_files = st.file_uploader(
        "📎 Upload files for comparison context (optional)",
        accept_multiple_files=True,
        type=None,
        key="compare_files",
    )

    if st.button(
        "Compare", type="primary", disabled=not (epic_1 and epic_2)
    ):
        clean_key_1 = _extract_epic_key(epic_1)
        clean_key_2 = _extract_epic_key(epic_2)
        cmp_status = st.status(
            f"Comparing {clean_key_1} vs {clean_key_2}...", expanded=True
        )
        cmp_bar = cmp_status.progress(0.0)
        st.session_state.compare_progress_log = []
        cmp_log_container = cmp_status.empty()

        def _on_cmp(msg: str, pct: float) -> None:
            st.session_state.compare_progress_log.append(f"⏳ {msg}")
            cmp_log_container.markdown(
                "\n".join(
                    f"- {line}" for line in st.session_state.compare_progress_log
                )
            )
            cmp_bar.progress(min(pct, 1.0))
            cmp_status.update(label=msg)

        try:
            result = bundle["session"].run_comparator(
                clean_key_1,
                clean_key_2,
                goal.strip() or None,
                context_options=compare_context,
                progress=_on_cmp,
            )
            cmp_status.update(
                label=f"✅ Comparison complete — "
                      f"{len(st.session_state.compare_progress_log)} steps",
                state="complete",
                expanded=True,
            )
        except Exception as e:
            cmp_status.update(
                label=f"❌ Comparison failed: {e}",
                state="error",
                expanded=True,
            )
            st.code(traceback.format_exc())
            result = None

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

                if compare_files:
                    st.markdown("---")
                    st.markdown("### 📎 Uploaded file analysis")
                    try:
                        from llm_client import get_backend as _get_be2
                        _be2 = _get_be2()
                        cmp_file_dicts = [
                            {
                                "name": uf.name,
                                "media_type": uf.type or "application/octet-stream",
                                "data": uf.getvalue(),
                            }
                            for uf in compare_files
                        ]
                        cmp_summary = json.dumps(result, default=str)[:30_000]
                        cmp_file_result = _be2.complete_with_files(
                            system=(
                                "You are an expert analyst. You have "
                                "comparison results for two Jira epics and "
                                "uploaded files. Integrate the file contents "
                                "with the comparison. For images describe "
                                "what you see. For data files analyze the "
                                "data. Relate everything to the comparison."
                            ),
                            text_prompt=(
                                f"Comparison of {clean_key_1} vs {clean_key_2}:\n\n"
                                f"{cmp_summary}\n\n"
                                f"The user uploaded {len(cmp_file_dicts)} file(s). "
                                f"Analyze them and integrate with the comparison."
                            ),
                            files=cmp_file_dicts,
                        )
                        st.markdown(cmp_file_result)
                    except Exception as e:
                        st.error(f"File analysis failed: {e}")
                        st.code(traceback.format_exc())

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

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
import traceback
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

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
        "jira_host": "https://healthtap.atlassian.net",
        "jira_email": "",
        "jira_token": "",
        "anthropic_key": "",
        "model": "claude-sonnet-4-5",
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


def _save_secrets_to_disk(values: Dict[str, str]) -> None:
    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Saved locally by the Streamlit UI. Gitignored — never commit.",
    ]
    for k, v in values.items():
        escaped = v.replace('"', '\\"')
        lines.append(f'{k} = "{escaped}"')
    SECRETS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _init_session_state() -> None:
    if "_creds_loaded" in st.session_state:
        return
    defaults = _load_saved_secrets()
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.session_state._creds_loaded = True


_init_session_state()


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
st.session_state.anthropic_key = st.sidebar.text_input(
    "Anthropic API key",
    value=st.session_state.anthropic_key,
    type="password",
    help="Generate at https://console.anthropic.com/settings/keys",
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
            "anthropic_key": st.session_state.anthropic_key,
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
os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_key or ""
os.environ["CLAUDE_MODEL"] = st.session_state.model or "claude-sonnet-4-5"


def _credentials_ok() -> bool:
    return all([
        st.session_state.jira_host,
        st.session_state.jira_email,
        st.session_state.jira_token,
        st.session_state.anthropic_key,
    ])


# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("🧠 Jira RAG Agent")
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
    from anthropic import Anthropic
    from agent_cli import AgentSession, SYSTEM_PROMPT, TOOLS, run_turn
except Exception as e:  # pragma: no cover
    st.error(f"Failed to import agent modules: {e}")
    st.code(traceback.format_exc())
    st.stop()


@st.cache_resource(show_spinner=False)
def get_session_bundle(jira_host: str, jira_email: str, anthropic_key: str):
    """Bundle of long-lived state.

    Caching on (jira_host, jira_email, anthropic_key) means the session is
    rebuilt when credentials change but reused across reruns otherwise.
    """
    session = AgentSession()
    client = Anthropic(api_key=anthropic_key)
    return {"session": session, "client": client, "history": []}


try:
    bundle = get_session_bundle(
        st.session_state.jira_host,
        st.session_state.jira_email,
        st.session_state.anthropic_key,
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
tab_chat, tab_analyze, tab_compare = st.tabs(
    ["💬 Chat", "🔍 Analyze Epic", "⚖️ Compare Epics"]
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
            with st.spinner("Thinking and fetching Jira data..."):
                try:
                    reply = run_turn(
                        bundle["client"],
                        bundle["session"],
                        bundle["history"],
                        prompt,
                    )
                except Exception as e:
                    reply = f"**Error:** {e}\n\n```\n{traceback.format_exc()}\n```"
            st.markdown(reply)
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

    if st.button("Analyze", type="primary", disabled=not epic_key):
        with st.spinner(
            f"Processing {epic_key} — first run can take a few minutes..."
        ):
            try:
                result = bundle["session"].run_analyzer(
                    epic_key.strip(),
                    focus,
                    custom_q.strip() or None,
                )
            except Exception as e:
                st.error(f"Analyzer failed: {e}")
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

    if st.button(
        "Compare", type="primary", disabled=not (epic_1 and epic_2)
    ):
        with st.spinner(f"Comparing {epic_1} vs {epic_2}..."):
            try:
                result = bundle["session"].run_comparator(
                    epic_1.strip(),
                    epic_2.strip(),
                    goal.strip() or None,
                )
            except Exception as e:
                st.error(f"Comparator failed: {e}")
                st.code(traceback.format_exc())
                result = None

        if result:
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"Compared {epic_1} vs {epic_2}.")
                col_l, col_r = st.columns(2)
                col_l.markdown(f"### {epic_1}")
                col_l.json(result.get("epic_1_summary", {}))
                col_r.markdown(f"### {epic_2}")
                col_r.json(result.get("epic_2_summary", {}))

                st.markdown("### Delta analysis")
                _render_generic(result.get("comparison", {}))

"""
Responsive Claude-powered CLI for the Jira RAG agent.

The two tools exposed to Claude are:
  1. `analyzer`   - run the full RAG analysis pipeline on a single epic
                    (test cases, edge cases, regression, bug detection,
                    detailed analysis, insights, recommendations).
  2. `comparator` - run the full pipeline on two epics, then generate a
                    delta analysis between them.

The user types natural language. Claude interprets the intent, picks the
right tool with the right arguments, we execute it, stream the tool
result back, and Claude then produces a polished response grounded in
the RAG output. This loop continues conversationally so users can
follow up ("what about regression for EPIC-2?", "compare it with EPIC-3"),
and each turn's AI processing is driven by the freshest tool output.

Usage:
    python agent_cli.py
    python agent_cli.py --prompt "Analyze epic PROJ-123 and list critical bugs"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from typing import Any, Dict, List

from config import config
from llm_client import get_backend
from rag_agent import JiraRAGAgent
from test_generator import TestCaseGenerator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("agent_cli")


# ---------------------------------------------------------------------------
# Tool definitions (given to Claude)
# ---------------------------------------------------------------------------
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "analyzer",
        "description": (
            "Analyze a single Jira Epic end-to-end. Fetches the epic and all "
            "linked tickets from Jira, indexes them into a vector store, and "
            "uses Claude to generate test cases, edge cases, regression "
            "analysis, bug detection, and optional insights or an action "
            "plan. Use this whenever the user asks to analyze, review, "
            "inspect, or reason about a single epic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "epic_key": {
                    "type": "string",
                    "description": "Jira epic key, e.g. 'PROJ-123'.",
                },
                "focus": {
                    "type": "string",
                    "enum": [
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
                    "description": (
                        "Which slice of the analysis the user cares about. "
                        "Pick the narrowest slice that answers their question; "
                        "'full' returns everything but is verbose."
                    ),
                },
                "custom_question": {
                    "type": "string",
                    "description": (
                        "Optional free-form question to run against the epic "
                        "using the RAG context (e.g. 'what are the data "
                        "integrity risks?'). Leave unset if not needed."
                    ),
                },
            },
            "required": ["epic_key", "focus"],
        },
    },
    {
        "name": "comparator",
        "description": (
            "Compare two Jira Epics side by side. Runs the analyzer on both "
            "if their results are not already cached, then produces a delta "
            "analysis highlighting scope differences, risk deltas, shared "
            "dependencies, and recommended actions. Use this whenever the "
            "user wants to compare, diff, or evaluate trade-offs between "
            "two epics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "epic_key_1": {"type": "string"},
                "epic_key_2": {"type": "string"},
                "comparison_goal": {
                    "type": "string",
                    "description": (
                        "What the user is trying to learn from the "
                        "comparison (e.g. 'which has higher regression "
                        "risk', 'overlap in scope'). Informs how we "
                        "summarise the result."
                    ),
                },
            },
            "required": ["epic_key_1", "epic_key_2"],
        },
    },
]


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the orchestrator for a Jira RAG agent. Your job is to help the
    user analyze and compare Jira epics by calling the `analyzer` and
    `comparator` tools. Always:

    1. Parse the user's request. If they mention one epic, use `analyzer`.
       If they mention two, use `comparator`.
    2. Pick the narrowest `focus` that satisfies their question so the tool
       result stays small and targeted. Only use `focus: "full"` when they
       explicitly ask for an exhaustive overview.
    3. After the tool result comes back, answer the user in clear prose
       grounded in the tool output. Cite ticket keys, bug ids, and risk
       levels verbatim. Do not invent data.
    4. If the user follows up, reuse cached results where possible rather
       than re-running expensive pipelines. Only re-run when the request
       needs fresh data or a different focus.
    5. If the user's request is ambiguous (missing epic key, unclear
       intent), ask a single concise clarifying question instead of
       guessing.
    """
).strip()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
class AgentSession:
    """Holds RAG agent state across a conversation.

    Caches processed epics so follow-up turns are fast. The same
    JiraRAGAgent instance is reused so the vector store stays hot.
    """

    def __init__(self) -> None:
        self._agent: JiraRAGAgent | None = None
        self._tg: TestCaseGenerator | None = None
        self._cache: Dict[str, Dict[str, Any]] = {}

    # Lazy init so `--help` etc. do not require API keys.
    @property
    def agent(self) -> JiraRAGAgent:
        if self._agent is None:
            self._agent = JiraRAGAgent()
        return self._agent

    @property
    def tg(self) -> TestCaseGenerator:
        if self._tg is None:
            self._tg = TestCaseGenerator()
        return self._tg

    def ensure_processed(self, epic_key: str) -> Dict[str, Any]:
        if epic_key not in self._cache:
            self._cache[epic_key] = self.agent.process_epic(epic_key)
        return self._cache[epic_key]

    # ------------------------------------------------------------------
    def run_analyzer(
        self,
        epic_key: str,
        focus: str,
        custom_question: str | None = None,
    ) -> Dict[str, Any]:
        results = self.ensure_processed(epic_key)
        if "error" in results:
            return {"error": results["error"], "epic_key": epic_key}

        epic_details = results.get("epic_details", {}) or {}
        tickets = results.get("tickets", {}) or {}

        if focus == "summary":
            out: Dict[str, Any] = {
                "epic_key": epic_key,
                "summary": epic_details.get("summary", ""),
                "status": epic_details.get("status", ""),
                "priority": epic_details.get("priority", ""),
                "ticket_count": len(tickets),
                "ticket_keys": list(tickets.keys()),
            }
        elif focus == "test_cases":
            out = {
                "epic_key": epic_key,
                "test_cases_by_ticket": {
                    k: t.get("test_cases", {}).get("test_cases", [])
                    for k, t in tickets.items()
                },
            }
        elif focus == "edge_cases":
            out = {
                "epic_key": epic_key,
                "edge_cases_by_ticket": {
                    k: t.get("edge_cases", {}).get("edge_cases", [])
                    for k, t in tickets.items()
                },
            }
        elif focus == "regression":
            out = {
                "epic_key": epic_key,
                "regression_by_ticket": {
                    k: t.get("regression_analysis", {})
                    for k, t in tickets.items()
                },
            }
        elif focus == "bug_analysis":
            out = {
                "epic_key": epic_key,
                "bug_analysis_by_ticket": {
                    k: t.get("bug_analysis", {}) for k, t in tickets.items()
                },
            }
        elif focus == "insights":
            out = {
                "epic_key": epic_key,
                "insights": self.tg.generate_data_insights(epic_details, results),
            }
        elif focus == "recommendations":
            issues: List[Dict[str, Any]] = []
            for t in tickets.values():
                for b in t.get("bug_analysis", {}).get("critical_bugs", []) or []:
                    issues.append({
                        "type": "Critical Bug",
                        "description": b.get("title", "Unknown bug"),
                    })
            out = {
                "epic_key": epic_key,
                "recommendations": self.tg.generate_recommendations(
                    epic_details, issues
                ),
            }
        elif focus == "action_plan":
            out = {
                "epic_key": epic_key,
                "action_plan": self.tg.generate_multi_action_plan(
                    epic_details, results
                ),
            }
        else:  # "full"
            out = {"epic_key": epic_key, "results": results}

        if custom_question:
            # Flatten data to run a single custom prompt through the LLM
            all_comments: List[Dict[str, Any]] = []
            all_changelog: List[Dict[str, Any]] = []
            for t in tickets.values():
                all_comments.extend(t.get("comments", []) or [])
                all_changelog.extend(t.get("changelog", []) or [])
            rag_ctx = self.agent.vector_store.search(custom_question)
            out["custom_answer"] = self.tg.custom_prompt_analysis(
                epic_details,
                all_comments,
                all_changelog,
                rag_ctx,
                custom_question,
            )
        return out

    # ------------------------------------------------------------------
    def run_comparator(
        self,
        epic_key_1: str,
        epic_key_2: str,
        comparison_goal: str | None = None,
    ) -> Dict[str, Any]:
        r1 = self.ensure_processed(epic_key_1)
        r2 = self.ensure_processed(epic_key_2)
        if "error" in r1 or "error" in r2:
            return {
                "error": "Failed to load one of the epics",
                "epic_1_error": r1.get("error"),
                "epic_2_error": r2.get("error"),
            }

        # Re-use the live vector store for RAG context
        ctx1 = self.agent.vector_store.search(
            f"epic {epic_key_1} {r1.get('epic_details', {}).get('summary', '')}"
        )
        ctx2 = self.agent.vector_store.search(
            f"epic {epic_key_2} {r2.get('epic_details', {}).get('summary', '')}"
        )
        comparison = self.tg.compare_epics(
            r1.get("epic_details", {}),
            r2.get("epic_details", {}),
            ctx1,
            ctx2,
        )

        return {
            "epic_key_1": epic_key_1,
            "epic_key_2": epic_key_2,
            "comparison_goal": comparison_goal or "",
            "comparison": comparison,
            "epic_1_summary": {
                "ticket_count": len(r1.get("tickets", {})),
                "status": r1.get("epic_details", {}).get("status", ""),
                "priority": r1.get("epic_details", {}).get("priority", ""),
            },
            "epic_2_summary": {
                "ticket_count": len(r2.get("tickets", {})),
                "status": r2.get("epic_details", {}).get("status", ""),
                "priority": r2.get("epic_details", {}).get("priority", ""),
            },
        }

    # ------------------------------------------------------------------
    def dispatch(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if tool_name == "analyzer":
                return self.run_analyzer(
                    epic_key=tool_input["epic_key"],
                    focus=tool_input.get("focus", "summary"),
                    custom_question=tool_input.get("custom_question"),
                )
            if tool_name == "comparator":
                return self.run_comparator(
                    epic_key_1=tool_input["epic_key_1"],
                    epic_key_2=tool_input["epic_key_2"],
                    comparison_goal=tool_input.get("comparison_goal"),
                )
            return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.exception("Tool %s failed", tool_name)
            return {"error": str(e)}


# ---------------------------------------------------------------------------
# Conversational loop with Claude tool-use
# ---------------------------------------------------------------------------
def _truncate_tool_result(data: Dict[str, Any], max_chars: int = 60_000) -> str:
    """Dump tool output to JSON and clamp so we never blow the model window."""
    text = json.dumps(data, default=str, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def run_turn(
    session: AgentSession,
    history: List[Dict[str, Any]],
    user_message: str,
    max_iterations: int = 6,
) -> str:
    """Run one user -> assistant turn, including any tool-use round-trips.

    Backend-agnostic: uses `llm_client.get_backend()` so it works for
    both the Anthropic-API flow (native tool_use) and the
    Claude-Code-subscription flow (JSON-routed tool_use).
    """
    backend = get_backend()
    history.append({"role": "user", "content": user_message})

    final_text_parts: List[str] = []

    for _ in range(max_iterations):
        decision = backend.step(SYSTEM_PROMPT, TOOLS, history)

        # Append whatever Claude emitted this step into history so the
        # next iteration sees it.
        if decision.api_assistant_content is not None:
            # API backend: preserve the raw content blocks so tool_use
            # ids line up with tool_result blocks we'll append later.
            history.append(
                {"role": "assistant", "content": decision.api_assistant_content}
            )
        elif decision.assistant_text or decision.tool_name:
            history.append(
                {
                    "role": "assistant",
                    "content": (
                        decision.assistant_text
                        or f"(requesting tool {decision.tool_name})"
                    ),
                }
            )

        if decision.assistant_text and decision.stop_reason != "tool_use":
            final_text_parts.append(decision.assistant_text)

        if decision.stop_reason != "tool_use" or not decision.tool_name:
            break

        # Execute the tool
        print(
            f"\n[tool] {decision.tool_name}({json.dumps(decision.tool_input)})",
            file=sys.stderr,
        )
        result = session.dispatch(decision.tool_name, decision.tool_input)
        result_text = _truncate_tool_result(result)

        # Append the tool result in the shape each backend expects
        if backend.name == "api":
            tool_use_id = getattr(decision, "_tool_use_id", None) or ""
            history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result_text,
                        }
                    ],
                }
            )
        else:
            history.append(
                {
                    "role": "user",
                    "content": (
                        f"TOOL_RESULT for {decision.tool_name}:\n{result_text}"
                    ),
                }
            )

    return "\n".join(final_text_parts).strip() or "(no response)"


def interactive(session: AgentSession) -> None:
    print(f"Jira RAG Agent - interactive mode (backend: {get_backend().name})")
    print("Type your request (or 'exit'/'quit' to leave).")
    history: List[Dict[str, Any]] = []
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user.lower() in {"exit", "quit", ":q"}:
            return
        try:
            reply = run_turn(session, history, user)
        except Exception as e:
            logger.exception("turn failed")
            print(f"[error] {e}")
            continue
        print(f"\nagent> {reply}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Responsive Claude-driven CLI for the Jira RAG agent."
    )
    parser.add_argument(
        "--prompt",
        help="Run a single prompt non-interactively and exit.",
    )
    args = parser.parse_args(argv)

    mode = config.CLAUDE_AUTH_MODE
    if mode == "api" and not config.ANTHROPIC_API_KEY:
        print(
            "ANTHROPIC_API_KEY is not set. Either export it, set "
            "CLAUDE_AUTH_MODE=subscription to use your Claude.ai plan, or "
            "enter credentials in the Streamlit sidebar.",
            file=sys.stderr,
        )
        return 2

    try:
        # Validate backend up front so we fail fast with a clear error.
        get_backend()
    except Exception as e:
        print(f"Backend init failed: {e}", file=sys.stderr)
        return 2

    session = AgentSession()

    if args.prompt:
        history: List[Dict[str, Any]] = []
        reply = run_turn(session, history, args.prompt)
        print(reply)
        return 0

    interactive(session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

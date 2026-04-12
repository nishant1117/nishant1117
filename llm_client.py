"""
Pluggable LLM backend for the Jira RAG Agent.

This module abstracts over *where* Claude calls are executed so the app
can run against either:

  1. Anthropic API directly (pay-per-token, requires an API key).
     Backend: `ApiKeyBackend`.
  2. Claude Code CLI, authenticated via OAuth against a Claude.ai
     subscription (Max / Pro). No API key, usage counts against the
     subscription's weekly quota. Backend: `SubscriptionBackend`.

Both backends implement the same tiny interface:

    class LLMBackend:
        def complete(system: str, user: str) -> str: ...
        def route_tool_call(system: str, user: str, tools: list[dict],
                            history: list[dict]) -> ToolDecision: ...

The first handles plain prompting (used by every method in
test_generator.py). The second lets the agent_cli tool-use loop ask
Claude to pick either the `analyzer` or `comparator` tool for a given
user message — on the API backend this uses Anthropic's native
tool_use; on the subscription backend it uses JSON-in-prompt routing
because the Agent SDK doesn't expose tool_use the same way.

The backend is selected via the env var `CLAUDE_AUTH_MODE`:

    CLAUDE_AUTH_MODE=api             # default, uses ANTHROPIC_API_KEY
    CLAUDE_AUTH_MODE=subscription    # uses claude-agent-sdk / Claude Code
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types shared by both backends
# ---------------------------------------------------------------------------
@dataclass
class ToolDecision:
    """Claude's decision about which tool (if any) to call."""

    tool_name: Optional[str] = None
    tool_input: Dict[str, Any] = field(default_factory=dict)
    assistant_text: str = ""            # any direct text Claude emitted
    stop_reason: str = "end_turn"       # "tool_use" | "end_turn"
    # Opaque history appendage for multi-turn conversations on the API
    # backend. The subscription backend ignores this.
    api_assistant_content: Any = None


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class LLMBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Single-turn prompt -> text."""

    @abstractmethod
    def step(
        self,
        system: str,
        tools: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> ToolDecision:
        """One step of the tool-use loop.

        Given system prompt, tool schemas, and message history, return
        a ToolDecision: either a tool_use (with tool_name + tool_input)
        or an end_turn with final assistant_text.

        Called once per loop iteration. The caller appends the decision
        to history, executes any requested tool, appends the tool
        result, and calls step() again until stop_reason == 'end_turn'.
        """


# ---------------------------------------------------------------------------
# Backend: Anthropic API (pay-per-token)
# ---------------------------------------------------------------------------
class ApiKeyBackend(LLMBackend):
    name = "api"

    def __init__(self) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run `pip install anthropic`."
            ) from e
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Enter it in the Streamlit "
                "sidebar or export it in your shell."
            )
        self._client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model = config.CLAUDE_MODEL

    def complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        out: List[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                out.append(block.text)
        return "\n".join(out).strip()

    def _call_with_tools(
        self,
        system: str,
        tools: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> ToolDecision:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=8192,
            system=system,
            tools=tools,
            messages=history,
        )
        decision = ToolDecision(
            stop_reason=str(getattr(resp, "stop_reason", "end_turn")),
            api_assistant_content=resp.content,
        )
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                decision.assistant_text += getattr(block, "text", "")
            elif btype == "tool_use":
                # First tool_use wins (we only orchestrate one per turn).
                if decision.tool_name is None:
                    decision.tool_name = getattr(block, "name", None)
                    decision.tool_input = dict(getattr(block, "input", {}) or {})
                    decision._tool_use_id = getattr(block, "id", None)  # type: ignore[attr-defined]
        return decision

    def step(
        self,
        system: str,
        tools: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> ToolDecision:
        return self._call_with_tools(system, tools, history)


# ---------------------------------------------------------------------------
# Backend: Claude Agent SDK (uses Claude Code OAuth against a Claude.ai sub)
# ---------------------------------------------------------------------------
def _extract_text_from_sdk_message(message: Any) -> str:
    """Best-effort text extraction from whatever shape the Agent SDK
    returns. The SDK emits AssistantMessage / UserMessage / etc. objects
    whose `content` is a list of blocks; each block usually has `.text`
    or a dict with key "text"."""
    out: List[str] = []
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                out.append(text)
    return "".join(out)


# The subscription backend uses JSON-in-prompt for tool routing because
# the Claude Agent SDK's query() API does not expose Anthropic-native
# tool_use blocks. We instruct Claude to output a specific JSON schema,
# then parse it. The same template handles both the initial routing
# decision and the post-tool-result summarization — Claude is told to
# answer from tool results if any are in history, else decide whether
# to call a tool.
_STEP_TEMPLATE = """\
You are the orchestrator for a Jira RAG agent.

Available tools:
{tools_json}

Conversation history (most recent last):
{history_text}

Your job: decide the NEXT step.

  - If the history already contains TOOL_RESULT entries that answer
    the user's latest question, respond with:

        {{"action": "final",
          "text": "<your final answer grounded in those tool results>"}}

    Cite ticket keys, bug ids, and risk levels verbatim from the tool
    results. Never invent data.

  - If a tool call is needed to answer the user's latest question,
    respond with EXACTLY this JSON shape (no markdown fences):

        {{"action": "tool_use",
          "tool_name": "<one of the tool names above>",
          "tool_input": {{ ... }}}}

    Choose the narrowest `focus` value that satisfies the user's intent.
    Only use "full" when the user explicitly asks for an exhaustive dump.
    Do not invent epic keys.

  - If the user's request is ambiguous (missing epic key, unclear
    intent), respond with:

        {{"action": "final",
          "text": "<one concise clarifying question>"}}

OUTPUT RULES:
  - Return ONLY the JSON. No markdown fences, no prose before or after.
  - The outer braces must be the first and last characters of your reply.
"""


def _history_to_text(history: List[Dict[str, Any]]) -> str:
    """Flatten the agent_cli-style history (which mixes strings, dicts,
    and tool_result blocks) into a plain-text transcript the SDK can
    read."""
    lines: List[str] = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if isinstance(content, str):
            lines.append(f"{role.upper()}: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_result":
                        lines.append(
                            f"TOOL_RESULT[{block.get('tool_use_id', '')}]: "
                            f"{block.get('content', '')}"
                        )
                    elif block.get("type") == "text":
                        lines.append(f"{role.upper()}: {block.get('text', '')}")
                elif hasattr(block, "text") and getattr(block, "text", None):
                    lines.append(f"{role.upper()}: {block.text}")
                elif hasattr(block, "name"):
                    lines.append(
                        f"{role.upper()} TOOL_USE: {block.name}"
                        f"({json.dumps(getattr(block, 'input', {}))})"
                    )
    return "\n".join(lines) if lines else "(empty)"


class SubscriptionBackend(LLMBackend):
    name = "subscription"

    def __init__(self) -> None:
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "claude-agent-sdk not installed. Run "
                "`pip install claude-agent-sdk` AND install Claude Code "
                "(`npm install -g @anthropic-ai/claude-code`), then run "
                "`claude /login` once to sign in with your Claude.ai account."
            ) from e
        self._model = config.CLAUDE_MODEL

    # ------------------------------------------------------------------
    async def _query_async(self, system: str, user: str) -> str:
        from claude_agent_sdk import query, ClaudeAgentOptions
        options = ClaudeAgentOptions(
            model=self._model,
            system_prompt=system,
            max_turns=1,
            permission_mode="bypassPermissions",
        )
        text_parts: List[str] = []
        try:
            async for message in query(prompt=user, options=options):
                text_parts.append(_extract_text_from_sdk_message(message))
        except Exception as e:
            logger.exception("Claude Agent SDK query failed")
            raise RuntimeError(
                f"Claude Agent SDK call failed: {e}. Make sure Claude Code "
                "is installed (`npm install -g @anthropic-ai/claude-code`) "
                "and you are logged in (`claude /login`)."
            ) from e
        return "".join(text_parts).strip()

    def _run_sync(self, system: str, user: str) -> str:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Streamlit's main thread isn't running an event loop,
                # but belt + suspenders for notebooks / async contexts.
                import nest_asyncio  # type: ignore
                nest_asyncio.apply()
                return loop.run_until_complete(self._query_async(system, user))
        except RuntimeError:
            pass
        return asyncio.run(self._query_async(system, user))

    # ------------------------------------------------------------------
    def complete(self, system: str, user: str) -> str:
        return self._run_sync(system, user)

    def step(
        self,
        system: str,
        tools: List[Dict[str, Any]],
        history: List[Dict[str, Any]],
    ) -> ToolDecision:
        prompt = _STEP_TEMPLATE.format(
            tools_json=json.dumps(tools, indent=2),
            history_text=_history_to_text(history),
        )
        raw = self._run_sync(system, prompt)
        return self._parse_router_response(raw)

    # ------------------------------------------------------------------
    def _parse_router_response(self, raw: str) -> ToolDecision:
        text = raw.strip()
        # Strip ```json fences defensively
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip().rstrip("`").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Router JSON parse failed, treating as final text")
            return ToolDecision(stop_reason="end_turn", assistant_text=raw)

        action = parsed.get("action")
        if action == "tool_use":
            return ToolDecision(
                stop_reason="tool_use",
                tool_name=parsed.get("tool_name"),
                tool_input=parsed.get("tool_input", {}) or {},
            )
        return ToolDecision(
            stop_reason="end_turn",
            assistant_text=str(parsed.get("text", raw)),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_CACHED_BACKEND: Optional[LLMBackend] = None
_CACHED_KEY: Optional[tuple] = None


def get_backend(force: bool = False) -> LLMBackend:
    """Return the selected backend, creating it lazily. Cached on the
    (mode, api_key, model) tuple so credential changes in the Streamlit
    sidebar rebuild the client."""
    global _CACHED_BACKEND, _CACHED_KEY
    mode = (config.CLAUDE_AUTH_MODE or "api").lower()
    key = (mode, config.ANTHROPIC_API_KEY, config.CLAUDE_MODEL)
    if not force and _CACHED_BACKEND is not None and _CACHED_KEY == key:
        return _CACHED_BACKEND

    if mode == "subscription":
        backend: LLMBackend = SubscriptionBackend()
    else:
        backend = ApiKeyBackend()

    _CACHED_BACKEND = backend
    _CACHED_KEY = key
    logger.info("LLM backend: %s", backend.name)
    return backend

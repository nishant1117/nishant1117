"""
Main RAG agent orchestrator.

Given an epic key, the agent:
  1. Pulls epic + ticket data from Jira.
  2. Indexes everything into a Chroma vector store.
  3. For each ticket, retrieves top-k related context and hands it to
     Claude via TestCaseGenerator to produce test cases, edge cases,
     regression analysis, detailed analysis, and bug detection.
  4. Returns a single structured results dict and can format a report.
"""
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

from jira_client import JiraClient
from vector_store import VectorStore
from test_generator import TestCaseGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextOptions:
    """What extra Jira context to pull in alongside the main ticket
    fields when running the analyzer / comparator pipeline.

    All toggles default to on for the "obvious" context (subtasks,
    linked issues, comments, changelog). Attachments are off by default
    because downloading text content costs bandwidth and can surprise
    users.
    """
    include_subtasks: bool = True
    include_linked: bool = True
    include_comments: bool = True
    include_changelog: bool = True
    include_attachments: bool = False

    def cache_key(self) -> Tuple[bool, bool, bool, bool, bool]:
        return (
            self.include_subtasks,
            self.include_linked,
            self.include_comments,
            self.include_changelog,
            self.include_attachments,
        )

    def as_dict(self) -> Dict[str, bool]:
        return asdict(self)


DEFAULT_CONTEXT_OPTIONS = ContextOptions()


class JiraRAGAgent:
    def __init__(self):
        """Initialize RAG agent components"""
        self.jira_client = JiraClient()
        self.vector_store = VectorStore()
        self.test_generator = TestCaseGenerator()
        # Raw Jira data cache — populated once per epic, reused across
        # different context-option combinations.
        self._jira_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("RAG Agent initialized")

    def fetch_epic_data(self, epic_key: str) -> Dict[str, Any]:
        """Fetch raw Jira data for an epic — epic details, tickets,
        subtasks, linked issues, comments, changelog, and attachments.

        Cached per epic_key so re-running the analyzer with different
        context options reuses the same Jira fetch. To force a re-fetch
        (e.g. the epic was updated), call `invalidate(epic_key)` first.
        """
        if epic_key in self._jira_cache:
            return self._jira_cache[epic_key]

        logger.info(f"Fetching Jira data for epic: {epic_key}")
        epic_details = self.jira_client.get_epic_details(epic_key)
        tickets = self.jira_client.get_epic_issues(epic_key)
        logger.info(f"Found {len(tickets)} tickets for epic {epic_key}")

        # Eagerly load per-ticket context. This is cheap compared to
        # the LLM calls that follow, and it lets us swap context options
        # later without re-hitting Jira.
        for ticket in tickets:
            key = ticket.get("key", "")
            if not key:
                continue
            ticket["comments"] = self.jira_client.get_issue_comments(key)
            ticket["changelog"] = self.jira_client.get_issue_changelog(key)
            ticket["attachments"] = self.jira_client.get_issue_attachments(key)

        data = {
            "epic_key": epic_key,
            "epic_details": epic_details,
            "tickets": tickets,
        }
        self._jira_cache[epic_key] = data
        return data

    def invalidate(self, epic_key: Optional[str] = None) -> None:
        """Drop the Jira cache for one epic or everything."""
        if epic_key is None:
            self._jira_cache.clear()
        else:
            self._jira_cache.pop(epic_key, None)

    def process_epic(
        self,
        epic_key: str,
        context_options: Optional[ContextOptions] = None,
    ) -> Dict[str, Any]:
        """Main method to process an epic and generate analysis and test cases.

        Args:
            epic_key: Jira epic key.
            context_options: Which extra context sections to include
                when prompting Claude per ticket. Defaults to
                DEFAULT_CONTEXT_OPTIONS (everything except attachments).
        """
        opts = context_options or DEFAULT_CONTEXT_OPTIONS
        try:
            data = self.fetch_epic_data(epic_key)
            epic_details = data["epic_details"]
            tickets = data["tickets"]

            documents = self._prepare_documents(epic_details, tickets)
            self.vector_store.clear_collection()
            self.vector_store.add_documents(documents)

            results: Dict[str, Any] = {
                "epic_key": epic_key,
                "epic_details": epic_details,
                "context_options": opts.as_dict(),
                "tickets": {},
            }

            for ticket in tickets:
                logger.info(f"Processing ticket: {ticket.get('key', '')}")
                filtered = self._filter_ticket(ticket, opts)
                # Inject parent epic description so Claude always knows
                # the broader context this ticket lives in.
                filtered["parent_epic_description"] = epic_details.get(
                    "description", ""
                )
                results["tickets"][ticket["key"]] = self._process_ticket(filtered)

            logger.info(f"Completed processing epic {epic_key}")
            return results
        except Exception as e:
            logger.error(f"Failed to process epic {epic_key}: {str(e)}")
            return {"epic_key": epic_key, "error": str(e)}

    def _filter_ticket(
        self, ticket: Dict[str, Any], opts: ContextOptions
    ) -> Dict[str, Any]:
        """Return a copy of the ticket dict with optional context
        sections zeroed out per ContextOptions. The filtered dict is
        what gets passed to test_generator so prompts see only what
        the user asked for."""
        filtered = dict(ticket)
        if not opts.include_subtasks:
            filtered["subtasks"] = []
        if not opts.include_linked:
            filtered["linked_issues"] = []
        if not opts.include_comments:
            filtered["comments"] = []
        if not opts.include_changelog:
            filtered["changelog"] = []
        if not opts.include_attachments:
            filtered["attachments"] = []
        return filtered

    def _prepare_documents(
        self, epic_details: Dict[str, Any], tickets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare documents from epic and tickets for the vector store.

        Now indexes much richer content per ticket — descriptions,
        comments, subtask summaries, and attachment text — so the RAG
        retrieval step returns genuinely useful context to Claude.
        """
        documents: List[Dict[str, Any]] = []

        # --- Epic-level document ---
        epic_content = "\n".join([
            f"Epic: {epic_details.get('key', '')}",
            f"Summary: {epic_details.get('summary', '')}",
            f"Description: {epic_details.get('description', '')}",
            f"Status: {epic_details.get('status', '')}",
            f"Priority: {epic_details.get('priority', '')}",
            f"Created: {epic_details.get('created', '')}",
            f"Updated: {epic_details.get('updated', '')}",
        ]).strip()
        documents.append({
            "id": f"epic_{epic_details.get('key', 'unknown')}",
            "content": epic_content,
            "source": epic_details.get("key", ""),
            "type": "epic",
            "ticket_key": epic_details.get("key", ""),
        })

        # --- Per-ticket documents ---
        for ticket in tickets:
            tk = ticket.get("key", "unknown")

            # Core ticket document (description + acceptance criteria).
            content_parts = [
                f"Ticket: {tk}",
                f"Summary: {ticket.get('summary', '')}",
                f"Description: {ticket.get('description', '')}",
                f"Status: {ticket.get('status', '')}",
                f"Priority: {ticket.get('priority', '')}",
                f"Type: {ticket.get('issue_type', '')}",
                f"Acceptance Criteria: {ticket.get('acceptance_criteria', '')}",
            ]
            documents.append({
                "id": f"ticket_{tk}",
                "content": "\n".join(content_parts).strip(),
                "source": tk,
                "type": "ticket",
                "ticket_key": tk,
            })

            # Comments document (all comment bodies concatenated).
            comments = ticket.get("comments") or []
            if comments:
                comment_text = "\n\n".join(
                    f"[{c.get('author', '')} on {c.get('created', '')}]\n"
                    f"{c.get('body', '')}"
                    for c in comments
                )
                documents.append({
                    "id": f"comments_{tk}",
                    "content": f"Comments for {tk}:\n{comment_text}",
                    "source": tk,
                    "type": "comments",
                    "ticket_key": tk,
                })

            # Subtask descriptions.
            subtasks = ticket.get("subtasks") or []
            if subtasks:
                sub_text = "\n".join(
                    f"- {s.get('key', '')} [{s.get('status', '')}] "
                    f"{s.get('summary', '')}: "
                    f"{(s.get('description') or '')[:500]}"
                    for s in subtasks
                )
                documents.append({
                    "id": f"subtasks_{tk}",
                    "content": f"Subtasks for {tk}:\n{sub_text}",
                    "source": tk,
                    "type": "subtasks",
                    "ticket_key": tk,
                })

            # Attachment text content.
            attachments = ticket.get("attachments") or []
            att_texts = [
                f"[{a.get('filename', '')}]\n{a.get('text_content', '')}"
                for a in attachments
                if a.get("text_content")
            ]
            if att_texts:
                documents.append({
                    "id": f"attachments_{tk}",
                    "content": (
                        f"Attachment contents for {tk}:\n"
                        + "\n\n---\n\n".join(att_texts)
                    ),
                    "source": tk,
                    "type": "attachments",
                    "ticket_key": tk,
                })

        return documents

    def _process_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual ticket with analysis and test cases.

        `ticket` is the already-filtered dict from `_filter_ticket`, so
        whichever of subtasks / linked_issues / comments / changelog /
        attachments are empty reflect the user's ContextOptions
        choices. We pull comments/changelog off the dict (they were
        fetched once in `fetch_epic_data`) and pass the full filtered
        ticket into the test generator so its prompts can cite
        subtasks, linked issues, and attachments when enabled.
        """
        try:
            comments = ticket.get("comments", []) or []
            changelog = ticket.get("changelog", []) or []

            query = f"{ticket.get('summary', '')} {ticket.get('description', '')}"
            context = self.vector_store.search(query)

            return {
                "ticket": ticket,
                "comments": comments,
                "changelog": changelog,
                "test_cases": self.test_generator.generate_test_cases(
                    ticket, context
                ),
                "edge_cases": self.test_generator.generate_edge_cases(
                    ticket, context
                ),
                "regression_analysis":
                    self.test_generator.generate_regression_analysis(
                        ticket, changelog, context
                    ),
                "analysis": self.test_generator.generate_detailed_analysis(
                    ticket, comments, changelog, context
                ),
                "bug_analysis":
                    self.test_generator.generate_bug_detection_analysis(
                        ticket, comments, changelog, context
                    ),
            }
        except Exception as e:
            logger.error(f"Failed to process ticket {ticket.get('key', '')}: {str(e)}")
            return {"ticket": ticket, "error": str(e)}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a formatted report from results"""
        lines: List[str] = []
        bar = "=" * 80
        sep = "─" * 80

        lines.append(bar)
        lines.append("JIRA EPIC ANALYSIS AND TEST CASE GENERATION REPORT")
        lines.append(bar)
        epic_details = results.get("epic_details", {}) or {}
        lines.append(f"\nEPIC: {results.get('epic_key', '')}")
        lines.append(f"Summary: {epic_details.get('summary', 'N/A')}")
        lines.append(f"Status: {epic_details.get('status', 'N/A')}")
        lines.append(f"Priority: {epic_details.get('priority', 'N/A')}")
        lines.append(f"\nTICKETS PROCESSED: {len(results.get('tickets', {}))}")
        lines.append("")

        for key, item in (results.get("tickets") or {}).items():
            lines.append(sep)
            lines.append(f"TICKET: {key}")
            lines.append(sep)
            lines.append(
                "Summary: " + item.get("ticket", {}).get("summary", "N/A")
            )
            lines.append(self._format_analysis(item.get("analysis", {})))
            lines.append("\nTEST CASES:")
            lines.append(self._format_test_cases(
                item.get("test_cases", {}).get("test_cases", [])
            ))
            lines.append("\nEDGE CASES:")
            lines.append(self._format_edge_cases(
                item.get("edge_cases", {}).get("edge_cases", [])
            ))
            lines.append("\nREGRESSION:")
            lines.append(self._format_regression_analysis(
                item.get("regression_analysis", {})
            ))
            lines.append("\nBUG ANALYSIS:")
            lines.append(self._format_bug_analysis(item.get("bug_analysis", {})))
            lines.append("")

        return "\n".join(lines)

    def _format_analysis(self, analysis: Dict[str, Any]) -> str:
        if not analysis:
            return "No analysis available"
        out = []
        if analysis.get("key_data_points"):
            out.append("Key Data Points:")
            for p in analysis["key_data_points"]:
                out.append(f"  • {p}")
        if analysis.get("requirements"):
            out.append("\nRequirements:")
            for r in analysis["requirements"]:
                out.append(f"  • {r}")
        if analysis.get("risks_and_concerns"):
            out.append("\nRisks and Concerns:")
            for r in analysis["risks_and_concerns"]:
                out.append(f"  • {r}")
        if analysis.get("insights"):
            out.append("\nInsights:")
            out.append(f"  {analysis['insights']}")
        return "\n".join(out)

    def _format_test_cases(self, test_cases: List[Dict[str, Any]]) -> str:
        if not test_cases:
            return "  No test cases generated"
        out = []
        for tc in test_cases:
            out.append(
                f"\n  [{tc.get('id', 'N/A')}] {tc.get('title', '')}"
            )
            out.append(f"    Priority: {tc.get('priority', 'N/A')}")
            out.append(f"    Category: {tc.get('category', 'N/A')}")
            out.append(f"    Expected Result: {tc.get('expected_result', 'N/A')}")
        return "\n".join(out)

    def _format_edge_cases(self, edge_cases: List[Dict[str, Any]]) -> str:
        if not edge_cases:
            return "  No edge cases identified"
        out = []
        for ec in edge_cases:
            out.append(
                f"\n  [{ec.get('id', 'N/A')}] {ec.get('title', '')}"
            )
            out.append(f"    Severity: {ec.get('severity', 'N/A')}")
            out.append(f"    Category: {ec.get('category', 'N/A')}")
            out.append(
                f"    Trigger Condition: {ec.get('trigger_condition', 'N/A')}"
            )
            out.append(
                f"    Expected Behavior: {ec.get('expected_behavior', 'N/A')}"
            )
        return "\n".join(out)

    def _format_regression_analysis(self, regression: Dict[str, Any]) -> str:
        if not regression:
            return "  No regression analysis available"
        out = [f"  Risk Level: {regression.get('regression_risk_level', 'N/A')}"]
        modules = regression.get("affected_modules") or []
        if modules:
            out.append("  Affected Modules:")
            for m in modules:
                out.append(
                    f"    • {m.get('module', '')} (Risk: {m.get('risk_level', '')})"
                )
                out.append(f"      {m.get('reason', '')}")
        features = regression.get("dependent_features") or []
        if features:
            out.append("  Dependent Features:")
            for f in features:
                out.append(
                    f"    • {f.get('feature', '')} ({f.get('dependency_type', '')})"
                )
                out.append(f"      Impact: {f.get('impact', '')}")
        return "\n".join(out)

    def _format_bug_analysis(self, bug_data: Dict[str, Any]) -> str:
        if not bug_data:
            return "  No bug analysis available"
        critical = bug_data.get("critical_bugs") or []
        if not critical:
            return "  No critical bugs identified"
        out = [f"  CRITICAL BUGS FOUND: {len(critical)}"]
        for b in critical:
            out.append(
                f"    [{b.get('bug_id', 'N/A')}] {b.get('title', '')} "
                f"({b.get('severity', '')})"
            )
            out.append(f"      Description: {b.get('description', '')}")
            out.append(f"      Trigger: {b.get('trigger_scenario', '')}")
            out.append(f"      Impact: {b.get('impact', '')}")
        return "\n".join(out)

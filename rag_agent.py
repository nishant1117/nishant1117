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
from typing import List, Dict, Any

from jira_client import JiraClient
from vector_store import VectorStore
from test_generator import TestCaseGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JiraRAGAgent:
    def __init__(self):
        """Initialize RAG agent components"""
        self.jira_client = JiraClient()
        self.vector_store = VectorStore()
        self.test_generator = TestCaseGenerator()
        logger.info("RAG Agent initialized")

    def process_epic(self, epic_key: str) -> Dict[str, Any]:
        """Main method to process an epic and generate analysis and test cases"""
        try:
            logger.info(f"Processing epic: {epic_key}")
            epic_details = self.jira_client.get_epic_details(epic_key)
            tickets = self.jira_client.get_epic_issues(epic_key)
            logger.info(f"Found {len(tickets)} tickets for epic {epic_key}")

            documents = self._prepare_documents(epic_details, tickets)
            self.vector_store.clear_collection()
            self.vector_store.add_documents(documents)

            results: Dict[str, Any] = {
                "epic_key": epic_key,
                "epic_details": epic_details,
                "tickets": {},
            }

            for ticket in tickets:
                logger.info(f"Processing ticket: {ticket.get('key', '')}")
                results["tickets"][ticket["key"]] = self._process_ticket(ticket)

            logger.info(f"Completed processing epic {epic_key}")
            return results
        except Exception as e:
            logger.error(f"Failed to process epic {epic_key}: {str(e)}")
            return {"epic_key": epic_key, "error": str(e)}

    def _prepare_documents(
        self, epic_details: Dict[str, Any], tickets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare documents from epic and tickets for vector store"""
        documents: List[Dict[str, Any]] = []

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

        for ticket in tickets:
            content_parts = [
                f"Ticket: {ticket.get('key', '')}",
                f"Summary: {ticket.get('summary', '')}",
                f"Description: {ticket.get('description', '')}",
                f"Status: {ticket.get('status', '')}",
                f"Priority: {ticket.get('priority', '')}",
                f"Type: {ticket.get('issue_type', '')}",
                f"Acceptance Criteria: {ticket.get('acceptance_criteria', '')}",
            ]
            documents.append({
                "id": f"ticket_{ticket.get('key', 'unknown')}",
                "content": "\n".join(content_parts).strip(),
                "source": ticket.get("key", ""),
                "type": "ticket",
                "ticket_key": ticket.get("key", ""),
            })

        return documents

    def _process_ticket(self, ticket: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual ticket with analysis and test cases"""
        try:
            key = ticket.get("key", "")
            comments = self.jira_client.get_issue_comments(key)
            changelog = self.jira_client.get_issue_changelog(key)

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

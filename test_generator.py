"""
Test case + analysis generation using RAG and Claude.

This module holds all the LLM-facing prompting logic. Every public method
takes already-fetched Jira data (plus optional RAG context) and returns
structured JSON by asking Claude to emit JSON and then parsing it.

The two high-level capabilities the agent exposes as tools are:
  - analyzer  -> generate_* methods on a single ticket / epic
  - comparator -> compare_epics / generate_impact_analysis across epics
"""
import json
import logging
from typing import List, Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import config

logger = logging.getLogger(__name__)


def _strip_json_fence(text: str) -> str:
    """Strip ```json ... ``` fences that Claude sometimes emits."""
    if not text:
        return ""
    t = text.strip()
    if "```json" in t:
        t = t.split("```json", 1)[1]
        t = t.split("```", 1)[0]
    elif t.startswith("```"):
        t = t.split("```", 1)[1]
        t = t.split("```", 1)[0]
    return t.strip()


class TestCaseGenerator:
    def __init__(self):
        """Initialize LLM for test case generation"""
        self.llm = ChatAnthropic(
            model=config.CLAUDE_MODEL,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            max_tokens=4096,
        )

    # ------------------------------------------------------------------
    # Core generation methods
    # ------------------------------------------------------------------
    def generate_test_cases(
        self, ticket_data: Dict[str, Any], context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate test cases for a ticket using RAG context"""
        try:
            context_text = "\n".join(
                c.get("content", "") for c in context
            ) or "No additional context available"
            prompt = self._create_test_case_prompt(ticket_data, context_text)
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert QA engineer specializing in test case "
                        "creation. Create comprehensive, detailed, and practical "
                        "test cases."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            parsed = self._parse_test_cases(response.content)
            logger.info(
                f"Generated {len(parsed.get('test_cases', []))} test cases for "
                f"{ticket_data.get('key', 'unknown')}"
            )
            return parsed
        except Exception as e:
            logger.error(f"Failed to generate test cases: {str(e)}")
            return {"test_cases": [], "error": str(e)}

    def generate_detailed_analysis(
        self,
        ticket_data: Dict[str, Any],
        comments: List[Dict[str, Any]],
        changelog: List[Dict[str, Any]],
        context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate detailed analysis of ticket including data extraction"""
        try:
            context_text = "\n".join(
                c.get("content", "") for c in context
            ) or "No additional context available"
            prompt = self._create_analysis_prompt(
                ticket_data, comments, changelog, context_text
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert software analyst. Provide comprehensive "
                        "analysis including key data points, requirements, test "
                        "scenarios, and risk assessment."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            parsed = self._parse_analysis(response.content)
            logger.info(
                f"Generated detailed analysis for "
                f"{ticket_data.get('key', 'unknown')}"
            )
            return parsed
        except Exception as e:
            logger.error(f"Failed to generate analysis: {str(e)}")
            return {"error": str(e)}

    def generate_edge_cases(
        self, ticket_data: Dict[str, Any], context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive edge cases for a ticket"""
        try:
            context_text = "\n".join(
                c.get("content", "") for c in context
            ) or "No additional context available"
            prompt = self._create_edge_case_prompt(ticket_data, context_text)
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert QA engineer specializing in edge case "
                        "identification and boundary testing. Identify all critical "
                        "edge cases and boundary conditions."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            parsed = self._parse_edge_cases(response.content)
            logger.info(
                f"Generated {len(parsed.get('edge_cases', []))} edge cases for "
                f"{ticket_data.get('key', 'unknown')}"
            )
            return parsed
        except Exception as e:
            logger.error(f"Failed to generate edge cases: {str(e)}")
            return {"edge_cases": [], "error": str(e)}

    def generate_regression_analysis(
        self,
        ticket_data: Dict[str, Any],
        changelog: List[Dict[str, Any]],
        context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate regression test analysis and impact assessment"""
        try:
            context_text = "\n".join(
                c.get("content", "") for c in context
            ) or "No additional context available"
            prompt = self._create_regression_prompt(
                ticket_data, changelog, context_text
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert regression testing specialist. Analyze "
                        "potential regression risks, affected areas, and required "
                        "regression test coverage."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            parsed = self._parse_regression_analysis(response.content)
            logger.info(
                f"Generated regression analysis for "
                f"{ticket_data.get('key', 'unknown')}"
            )
            return parsed
        except Exception as e:
            logger.error(f"Failed to generate regression analysis: {str(e)}")
            return {"error": str(e)}

    def generate_bug_detection_analysis(
        self,
        ticket_data: Dict[str, Any],
        comments: List[Dict[str, Any]],
        changelog: List[Dict[str, Any]],
        context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate bug detection and vulnerability analysis"""
        try:
            context_text = "\n".join(
                c.get("content", "") for c in context
            ) or "No additional context available"
            prompt = self._create_bug_detection_prompt(
                ticket_data, comments, changelog, context_text
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert bug detection specialist and security "
                        "researcher. Identify potential bugs, vulnerabilities, and "
                        "issues based on dependencies and test scenarios."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            parsed = self._parse_bug_analysis(response.content)
            logger.info(
                f"Generated bug detection analysis for "
                f"{ticket_data.get('key', 'unknown')}"
            )
            return parsed
        except Exception as e:
            logger.error(f"Failed to generate bug detection analysis: {str(e)}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Comparator capability
    # ------------------------------------------------------------------
    def compare_epics(
        self,
        epic1_data: Dict[str, Any],
        epic2_data: Dict[str, Any],
        context1: List[Dict[str, Any]],
        context2: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare two epics and generate delta analysis"""
        try:
            ctx1 = "\n".join(c.get("content", "") for c in context1) or \
                "No additional context available"
            ctx2 = "\n".join(c.get("content", "") for c in context2) or \
                "No additional context available"
            prompt = self._create_comparison_prompt(
                epic1_data, epic2_data, ctx1, ctx2
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert software architect and project manager. "
                        "Provide comprehensive comparison analysis between two epics "
                        "with delta analysis, risk assessment, and recommendations."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            parsed = self._parse_comparison_analysis(response.content)
            logger.info(
                f"Generated comparison analysis for "
                f"{epic1_data.get('key', '?')} vs {epic2_data.get('key', '?')}"
            )
            return parsed
        except Exception as e:
            logger.error(f"Failed to generate comparison analysis: {str(e)}")
            return {"error": str(e)}

    def custom_prompt_analysis(
        self,
        ticket_data: Dict[str, Any],
        comments: List[Dict[str, Any]],
        changelog: List[Dict[str, Any]],
        context: List[Dict[str, Any]],
        custom_prompt: str,
    ) -> Dict[str, Any]:
        """Generate analysis based on custom user prompt"""
        try:
            ctx = "\n".join(c.get("content", "") for c in context) or \
                "No additional context available"
            comment_text = "\n".join(
                f"- {c.get('author', '')}: {c.get('body', '')}" for c in comments
            ) or "No comments"
            changelog_text = "\n".join(
                f"- {h.get('field', '')}: {h.get('from_value', '')} -> "
                f"{h.get('to_value', '')}" for h in changelog
            ) or "No changelog"

            ticket_block = (
                f"\nTicket: {ticket_data.get('key', '')} - "
                f"{ticket_data.get('summary', '')}"
                f"\nDescription: {ticket_data.get('description', 'N/A')}"
                f"\nStatus: {ticket_data.get('status', 'N/A')}"
                f"\nPriority: {ticket_data.get('priority', 'N/A')}"
                f"\nAssignee: {ticket_data.get('assignee', 'N/A')}"
                f"\n\nComments:\n{comment_text}"
                f"\n\nChange History:\n{changelog_text}"
                f"\n\nRAG Context:\n{ctx}"
            )

            messages = [
                SystemMessage(
                    content=(
                        "You are a helpful software engineering analyst. Answer the "
                        "user's question about the ticket using only the provided "
                        "data. Return JSON when structured output is useful, "
                        "otherwise plain prose."
                    )
                ),
                HumanMessage(
                    content=f"{custom_prompt}\n\n{ticket_block}"
                ),
            ]
            response = self.llm.invoke(messages)
            parsed = self._parse_custom_analysis(response.content)
            logger.info(
                f"Generated custom analysis for {ticket_data.get('key', 'unknown')}"
            )
            return parsed
        except Exception as e:
            logger.error(f"Failed to generate custom analysis: {str(e)}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Higher-level Claude "thinking" helpers
    # ------------------------------------------------------------------
    def generate_data_insights(
        self, epic_data: Dict[str, Any], analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate Claude AI insights from epic data and analysis results"""
        try:
            prompt = (
                "\nAnalyze the following epic data and analysis results to provide "
                "comprehensive insights:\n\n"
                f"EPIC INFORMATION:\n  Key: {epic_data.get('key', '')}"
                f"\n  Summary: {epic_data.get('summary', '')}"
                f"\n  Description: {epic_data.get('description', 'N/A')}"
                f"\n  Status: {epic_data.get('status', 'N/A')}"
                f"\n  Priority: {epic_data.get('priority', 'N/A')}"
                f"\n\nTICKETS: {len(analysis_results.get('tickets', {}))}"
                f"\n\nANALYSIS RESULTS:\n"
                f"  Total Test Cases: {sum(len(t.get('test_cases', {}).get('test_cases', [])) for t in analysis_results.get('tickets', {}).values())}"
                "\n\nProvide insights as JSON with keys: key_observations, "
                "quality_risks, coverage_gaps, strategic_recommendations."
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are a senior engineering insights analyst. Surface "
                        "the non-obvious patterns in the data."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            return self._parse_insights(response.content)
        except Exception as e:
            logger.error(f"Failed to generate insights: {str(e)}")
            return {"error": str(e)}

    def generate_recommendations(
        self, epic_data: Dict[str, Any], issues: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate Claude AI recommendations for addressing issues"""
        try:
            issues_text = "\n".join(
                f"- {i.get('type', 'Issue')}: {i.get('description', 'N/A')}"
                for i in issues
            ) or "No issues identified"
            prompt = (
                f"\nFor the following epic and identified issues, provide detailed "
                f"recommendations:\n\n"
                f"EPIC: {epic_data.get('key', '')} - {epic_data.get('summary', '')}"
                f"\n\nIDENTIFIED ISSUES:\n{issues_text}"
                "\n\nFor each issue, provide:\n"
                "1. Root cause analysis\n"
                "2. Immediate action items\n"
                "3. Long-term solutions\n"
                "4. Risk if not addressed\n"
                "5. Effort estimate to fix\n"
                "6. Success criteria\n\n"
                "Also provide:\n"
                "- Prioritized action plan\n"
                "- Dependencies between fixes\n"
                "- Quick wins vs long-term improvements\n"
                "- Resource requirements\n\n"
                "Return as JSON with structured recommendations."
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are an experienced software engineer and consultant. "
                        "Provide practical, actionable recommendations that teams "
                        "can implement."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            logger.info("Generated recommendations")
            return self._parse_recommendations(response.content)
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {str(e)}")
            return {"error": str(e)}

    def generate_multi_action_plan(
        self, epic_data: Dict[str, Any], analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a comprehensive multi-action plan orchestrated by Claude AI"""
        try:
            ticket_count = len(analysis_results.get("tickets", {}))
            prompt = (
                f"\nCreate a comprehensive, multi-step action plan for this epic:\n\n"
                f"EPIC: {epic_data.get('key', '')} - {epic_data.get('summary', '')}"
                f"\nTICKETS: {ticket_count}"
                f"\nSTATUS: {epic_data.get('status', '')}"
                "\n\nCreate a multi-action orchestration plan with:\n"
                "1. PHASE 1: PREPARATION (Pre-development)\n"
                "2. PHASE 2: DEVELOPMENT (Core work)\n"
                "3. PHASE 3: TESTING (Quality assurance)\n"
                "4. PHASE 4: DEPLOYMENT & MONITORING\n"
                "5. RISK MITIGATION ACTIONS\n"
                "6. SUCCESS METRICS\n\n"
                "Return as JSON with detailed, time-ordered action steps."
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are a project orchestration expert. Create detailed, "
                        "executable action plans that teams can follow."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            return self._parse_action_plan(response.content)
        except Exception as e:
            logger.error(f"Failed to generate action plan: {str(e)}")
            return {"error": str(e)}

    def generate_impact_analysis(
        self, epic_data: Dict[str, Any], changes: List[str]
    ) -> Dict[str, Any]:
        """Generate impact analysis for changes"""
        try:
            changes_text = "\n".join(f"- {c}" for c in changes) or \
                "No specific changes provided"
            ticket_count = len(epic_data.get("tickets", []) or [])
            prompt = (
                f"\nAnalyze the impact of these changes in the context of the epic:"
                f"\n\nEPIC: {epic_data.get('key', '')} - "
                f"{epic_data.get('summary', '')}"
                f"\n\nCHANGES:\n{changes_text}"
                f"\n\nCURRENT TICKETS: {ticket_count}"
                "\n\nProvide comprehensive impact analysis as JSON with: "
                "direct_impact, ripple_effects, risk_implications, "
                "testing_implications, deployment_considerations, stakeholder_impact."
            )
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert in impact analysis. Provide thorough, "
                        "thoughtful analysis of how changes affect systems."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            logger.info("Generated impact analysis")
            return self._parse_impact_analysis(response.content)
        except Exception as e:
            logger.error(f"Failed to generate impact analysis: {str(e)}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _create_test_case_prompt(
        self, ticket_data: Dict[str, Any], context: str
    ) -> str:
        return (
            "\nBased on the following ticket information and context, generate "
            "comprehensive test cases.\n\n"
            "TICKET INFORMATION:\n"
            f"- Key: {ticket_data.get('key', 'N/A')}\n"
            f"- Summary: {ticket_data.get('summary', 'N/A')}\n"
            f"- Description: {ticket_data.get('description', 'N/A')}\n"
            f"- Acceptance Criteria: {ticket_data.get('acceptance_criteria', 'N/A')}\n"
            f"- Issue Type: {ticket_data.get('issue_type', 'N/A')}\n"
            f"- Priority: {ticket_data.get('priority', 'N/A')}\n\n"
            f"CONTEXT FROM RAG:\n{context}\n\n"
            f"Generate {config.MIN_TEST_CASES_PER_TICKET} to "
            f"{config.MAX_TEST_CASES_PER_TICKET} test cases in the following "
            "JSON format:\n"
            '{\n'
            '    "test_cases": [\n'
            '        {\n'
            '            "id": "TC-001",\n'
            '            "title": "Test case title",\n'
            '            "description": "Detailed description",\n'
            '            "preconditions": ["precondition 1", "precondition 2"],\n'
            '            "steps": ["step 1", "step 2", "step 3"],\n'
            '            "expected_result": "Expected outcome",\n'
            '            "priority": "High/Medium/Low",\n'
            '            "category": "Functional/Edge Case/Security/Performance"\n'
            '        }\n'
            '    ]\n'
            '}\n\nOnly return valid JSON, no additional text.\n'
        )

    def _create_analysis_prompt(
        self,
        ticket_data: Dict[str, Any],
        comments: List[Dict[str, Any]],
        changelog: List[Dict[str, Any]],
        context: str,
    ) -> str:
        comment_text = "\n".join(
            f"- {c.get('author', '')}: {c.get('body', '')}" for c in comments
        ) or "No comments"
        changelog_text = "\n".join(
            f"- {h.get('field', '')}: {h.get('from_value', '')} -> "
            f"{h.get('to_value', '')}" for h in changelog
        ) or "No changelog"
        labels = ", ".join(ticket_data.get("labels", []) or [])

        return (
            "\nProvide a comprehensive analysis of the following Jira ticket "
            "including extracted data, test scenarios, and insights.\n\n"
            "TICKET INFORMATION:\n"
            f"- Key: {ticket_data.get('key', 'N/A')}\n"
            f"- Summary: {ticket_data.get('summary', 'N/A')}\n"
            f"- Description: {ticket_data.get('description', 'N/A')}\n"
            f"- Acceptance Criteria: {ticket_data.get('acceptance_criteria', 'N/A')}\n"
            f"- Issue Type: {ticket_data.get('issue_type', 'N/A')}\n"
            f"- Priority: {ticket_data.get('priority', 'N/A')}\n"
            f"- Status: {ticket_data.get('status', 'N/A')}\n"
            f"- Labels: {labels}\n\n"
            f"COMMENTS:\n{comment_text}\n\n"
            f"CHANGELOG:\n{changelog_text}\n\n"
            f"RAG CONTEXT:\n{context}\n\n"
            "Return analysis as JSON with keys: key_data_points, requirements, "
            "risks_and_concerns, insights, test_scenarios.\n"
        )

    def _create_edge_case_prompt(
        self, ticket_data: Dict[str, Any], context: str
    ) -> str:
        return (
            "\nIdentify and generate comprehensive edge cases and boundary "
            "conditions for the following ticket.\n\n"
            "TICKET INFORMATION:\n"
            f"- Key: {ticket_data.get('key', 'N/A')}\n"
            f"- Summary: {ticket_data.get('summary', 'N/A')}\n"
            f"- Description: {ticket_data.get('description', 'N/A')}\n"
            f"- Acceptance Criteria: {ticket_data.get('acceptance_criteria', 'N/A')}\n"
            f"- Issue Type: {ticket_data.get('issue_type', 'N/A')}\n\n"
            f"CONTEXT FROM RAG:\n{context}\n\n"
            "Generate edge cases in the following JSON format:\n"
            '{\n'
            '    "edge_cases": [\n'
            '        {\n'
            '            "id": "EC-001",\n'
            '            "title": "Edge case title",\n'
            '            "description": "Detailed description of the edge case",\n'
            '            "trigger_condition": "Condition that triggers this edge case",\n'
            '            "boundary_values": ["value 1", "value 2"],\n'
            '            "expected_behavior": "How the system should behave",\n'
            '            "severity": "High/Medium/Low",\n'
            '            "category": "Boundary/Null/Empty/Overflow/Timeout/Concurrency/Data Validation"\n'
            '        }\n'
            '    ],\n'
            '    "critical_scenarios": ["Scenario 1"],\n'
            '    "boundary_test_values": ["Value 1"]\n'
            '}\n\nOnly return valid JSON, no additional text.\n'
        )

    def _create_regression_prompt(
        self,
        ticket_data: Dict[str, Any],
        changelog: List[Dict[str, Any]],
        context: str,
    ) -> str:
        changelog_text = "\n".join(
            f"- {h.get('field', '')}: {h.get('from_value', '')} -> "
            f"{h.get('to_value', '')}" for h in changelog
        ) or "No changelog"
        return (
            "\nPerform a comprehensive regression testing analysis for the "
            "following ticket.\n\n"
            "TICKET INFORMATION:\n"
            f"- Key: {ticket_data.get('key', 'N/A')}\n"
            f"- Summary: {ticket_data.get('summary', 'N/A')}\n"
            f"- Description: {ticket_data.get('description', 'N/A')}\n"
            f"- Type: {ticket_data.get('issue_type', 'N/A')}\n"
            f"- Status: {ticket_data.get('status', 'N/A')}\n\n"
            f"CHANGELOG:\n{changelog_text}\n\n"
            f"RELATED CONTEXT:\n{context}\n\n"
            "Generate regression analysis in the following JSON format:\n"
            '{\n'
            '    "regression_risk_level": "High/Medium/Low",\n'
            '    "affected_modules": [\n'
            '        {"module": "Module name", "risk_level": "High/Medium/Low", "reason": "Why"}\n'
            '    ],\n'
            '    "dependent_features": [\n'
            '        {"feature": "Feature name", "dependency_type": "Direct/Indirect", "impact": "Impact description"}\n'
            '    ],\n'
            '    "regression_test_areas": ["Area 1"],\n'
            '    "regression_test_cases": [\n'
            '        {"id": "RTC-001", "description": "Test description", "focus_area": "Focus"}\n'
            '    ],\n'
            '    "recommendations": ["Recommendation 1"],\n'
            '    "risk_mitigation_strategies": ["Strategy 1"]\n'
            '}\n\nOnly return valid JSON, no additional text.\n'
        )

    def _create_bug_detection_prompt(
        self,
        ticket_data: Dict[str, Any],
        comments: List[Dict[str, Any]],
        changelog: List[Dict[str, Any]],
        context: str,
    ) -> str:
        comment_text = "\n".join(
            f"- {c.get('author', '')}: {c.get('body', '')}" for c in comments
        ) or "No comments"
        changelog_text = "\n".join(
            f"- {h.get('field', '')}: {h.get('from_value', '')} -> "
            f"{h.get('to_value', '')}" for h in changelog
        ) or "No changelog"
        return (
            "\nPerform a comprehensive bug detection and vulnerability analysis "
            "for the following ticket.\n\n"
            "TICKET INFORMATION:\n"
            f"- Key: {ticket_data.get('key', 'N/A')}\n"
            f"- Summary: {ticket_data.get('summary', 'N/A')}\n"
            f"- Description: {ticket_data.get('description', 'N/A')}\n"
            f"- Type: {ticket_data.get('issue_type', 'N/A')}\n"
            f"- Priority: {ticket_data.get('priority', 'N/A')}\n"
            f"- Status: {ticket_data.get('status', 'N/A')}\n\n"
            f"COMMENTS:\n{comment_text}\n\n"
            f"CHANGELOG:\n{changelog_text}\n\n"
            f"DEPENDENCIES AND RELATIONSHIPS:\n{context}\n\n"
            "Analyze and return a JSON object with keys: critical_bugs, "
            "potential_vulnerabilities, dependency_risks, integration_risks, "
            "error_handling_gaps, concurrency_issues, "
            "recommended_tests_to_prevent, high_risk_areas.\n\n"
            "Only return valid JSON, no additional text.\n"
        )

    def _create_comparison_prompt(
        self,
        epic1: Dict[str, Any],
        epic2: Dict[str, Any],
        context1: str,
        context2: str,
    ) -> str:
        def epic_block(epic):
            return (
                f"{epic.get('key', '')} - {epic.get('summary', '')}"
                f"\nDescription: {epic.get('description', 'N/A')}"
                f"\nStatus: {epic.get('status', 'N/A')}"
                f"\nPriority: {epic.get('priority', 'N/A')}"
                f"\nTickets: {len(epic.get('tickets', []) or [])}"
            )

        return (
            "\nCompare the following two epics and provide comprehensive delta "
            "analysis:\n\n"
            f"EPIC 1: {epic_block(epic1)}\n\n"
            f"Context for Epic 1:\n{context1}\n\n"
            f"EPIC 2: {epic_block(epic2)}\n\n"
            f"Context for Epic 2:\n{context2}\n\n"
            "Return JSON with keys: summary, scope_differences, "
            "risk_delta, shared_dependencies, recommended_actions, "
            "prioritization_guidance."
        )

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------
    def _safe_load(self, response_text: str, default: Any) -> Any:
        try:
            return json.loads(_strip_json_fence(response_text))
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {str(e)}")
            return default

    def _parse_test_cases(self, response_text: str) -> Dict[str, Any]:
        parsed = self._safe_load(response_text, None)
        if parsed is None:
            return {"test_cases": [], "raw": response_text}
        return {"test_cases": parsed.get("test_cases", [])}

    def _parse_analysis(self, response_text: str) -> Dict[str, Any]:
        return self._safe_load(response_text, {"raw": response_text})

    def _parse_edge_cases(self, response_text: str) -> Dict[str, Any]:
        parsed = self._safe_load(response_text, None)
        if parsed is None:
            return {"edge_cases": [], "raw": response_text}
        return parsed

    def _parse_regression_analysis(self, response_text: str) -> Dict[str, Any]:
        return self._safe_load(response_text, {"raw": response_text})

    def _parse_bug_analysis(self, response_text: str) -> Dict[str, Any]:
        return self._safe_load(response_text, {"raw": response_text})

    def _parse_comparison_analysis(self, response_text: str) -> Dict[str, Any]:
        parsed = self._safe_load(response_text, None)
        if parsed is None:
            return {"error": "Failed to parse response", "raw": response_text}
        return parsed

    def _parse_custom_analysis(self, response_text: str) -> Dict[str, Any]:
        parsed = self._safe_load(response_text, None)
        if parsed is None:
            return {"text": response_text}
        return parsed

    def _parse_insights(self, response_text: str) -> Dict[str, Any]:
        return self._safe_load(response_text, {"raw_insights": response_text})

    def _parse_recommendations(self, response_text: str) -> Dict[str, Any]:
        return self._safe_load(response_text, {"raw_recommendations": response_text})

    def _parse_action_plan(self, response_text: str) -> Dict[str, Any]:
        return self._safe_load(response_text, {"raw_plan": response_text})

    def _parse_impact_analysis(self, response_text: str) -> Dict[str, Any]:
        return self._safe_load(response_text, {"raw_analysis": response_text})

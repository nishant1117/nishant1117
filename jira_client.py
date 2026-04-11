"""
Jira client for fetching epic and ticket data.

Wraps the `jira` SDK and exposes the minimal surface the RAG agent needs:
epic details, epic issues, subtasks, linked issues, comments, and changelog.
All methods return plain dicts so they can be serialized into the vector store
or handed to Claude as tool context.
"""
import logging
from typing import List, Dict, Any

from jira import JIRA

from config import config

logger = logging.getLogger(__name__)


class JiraClient:
    def __init__(self):
        """Initialize Jira client"""
        self.client = JIRA(
            server=config.JIRA_HOST,
            basic_auth=(config.JIRA_EMAIL, config.JIRA_API_TOKEN),
        )
        logger.info(f"Connected to Jira: {config.JIRA_HOST}")

    def get_epic_details(self, epic_key: str) -> Dict[str, Any]:
        """Fetch epic details"""
        try:
            issue = self.client.issue(epic_key)
            return {
                "key": issue.key,
                "summary": getattr(issue.fields, "summary", "") or "",
                "description": getattr(issue.fields, "description", "") or "",
                "status": str(getattr(issue.fields.status, "name", "None")),
                "priority": str(
                    getattr(getattr(issue.fields, "priority", None), "name", "None")
                ),
                "created": str(getattr(issue.fields, "created", "")),
                "updated": str(getattr(issue.fields, "updated", "")),
            }
        except Exception as e:
            logger.error(f"Failed to fetch epic {epic_key}: {str(e)}")
            return {}

    def get_epic_issues(self, epic_key: str) -> List[Dict[str, Any]]:
        """Fetch all issues linked to an epic"""
        try:
            jql = f'"Epic Link" = {epic_key} OR key = {epic_key}'
            issues = self.client.search_issues(jql, maxResults=500)
            tickets = []
            for issue in issues:
                if issue.key == epic_key:
                    continue
                data = self._extract_issue_data(issue)
                data["subtasks"] = self.get_issue_subtasks(issue.key)
                data["linked_issues"] = self.get_linked_issues(issue.key)
                tickets.append(data)
            logger.info(f"Found {len(tickets)} tickets in epic {epic_key}")
            return tickets
        except Exception as e:
            logger.error(f"Failed to fetch issues for epic {epic_key}: {str(e)}")
            return []

    def get_issue_subtasks(self, issue_key: str) -> List[Dict[str, Any]]:
        """Fetch all subtasks for an issue"""
        try:
            issue = self.client.issue(issue_key)
            subtasks = []
            if hasattr(issue.fields, "subtasks"):
                for sub in issue.fields.subtasks:
                    subtasks.append({
                        "key": sub.key,
                        "summary": getattr(sub.fields, "summary", ""),
                        "status": str(getattr(sub.fields.status, "name", "None")),
                        "priority": str(
                            getattr(getattr(sub.fields, "priority", None), "name", "None")
                        ),
                        "assignee": str(
                            getattr(
                                getattr(sub.fields, "assignee", None),
                                "displayName",
                                "Unassigned",
                            )
                        ),
                    })
            logger.info(f"Found {len(subtasks)} subtasks for {issue_key}")
            return subtasks
        except Exception as e:
            logger.error(f"Failed to fetch subtasks for {issue_key}: {str(e)}")
            return []

    def get_linked_issues(self, issue_key: str) -> List[Dict[str, Any]]:
        """Fetch all linked issues for an issue"""
        try:
            issue = self.client.issue(issue_key, expand="changelog")
            linked = []
            if hasattr(issue.fields, "issuelinks"):
                for link in issue.fields.issuelinks:
                    if hasattr(link, "outwardIssue"):
                        other = link.outwardIssue
                        direction = "outward"
                        link_type = getattr(link.type, "outward", "")
                    elif hasattr(link, "inwardIssue"):
                        other = link.inwardIssue
                        direction = "inward"
                        link_type = getattr(link.type, "inward", "")
                    else:
                        continue
                    linked.append({
                        "key": other.key,
                        "summary": getattr(other.fields, "summary", ""),
                        "status": str(getattr(other.fields.status, "name", "None")),
                        "priority": str(
                            getattr(
                                getattr(other.fields, "priority", None),
                                "name",
                                "None",
                            )
                        ),
                        "link_type": link_type,
                        "direction": direction,
                    })
            logger.info(f"Found {len(linked)} linked issues for {issue_key}")
            return linked
        except Exception as e:
            logger.error(f"Failed to fetch linked issues for {issue_key}: {str(e)}")
            return []

    def get_issue_comments(self, issue_key: str) -> List[Dict[str, Any]]:
        """Fetch all comments from an issue"""
        try:
            issue = self.client.issue(issue_key)
            comments = []
            for c in issue.fields.comment.comments:
                comments.append({
                    "author": str(getattr(c.author, "displayName", "")),
                    "created": str(getattr(c, "created", "")),
                    "body": getattr(c, "body", "") or "",
                })
            return comments
        except Exception as e:
            logger.error(f"Failed to fetch comments for {issue_key}: {str(e)}")
            return []

    def get_issue_changelog(self, issue_key: str) -> List[Dict[str, Any]]:
        """Fetch changelog/history for an issue"""
        try:
            issue = self.client.issue(issue_key, expand="changelog")
            history = []
            if hasattr(issue, "changelog"):
                for h in issue.changelog.histories:
                    for item in h.items:
                        history.append({
                            "created": str(getattr(h, "created", "")),
                            "author": str(getattr(h.author, "displayName", "")),
                            "field": getattr(item, "field", ""),
                            "from_value": getattr(item, "fromString", "") or "",
                            "to_value": getattr(item, "toString", "") or "",
                        })
            return history
        except Exception as e:
            logger.error(f"Failed to fetch changelog for {issue_key}: {str(e)}")
            return []

    def _extract_issue_data(self, issue) -> Dict[str, Any]:
        """Extract relevant data from a Jira issue"""
        return {
            "key": issue.key,
            "summary": getattr(issue.fields, "summary", "") or "",
            "description": getattr(issue.fields, "description", "") or "",
            "status": str(getattr(issue.fields.status, "name", "None")),
            "priority": str(
                getattr(getattr(issue.fields, "priority", None), "name", "None")
            ),
            "issue_type": str(
                getattr(getattr(issue.fields, "issuetype", None), "name", "None")
            ),
            "assignee": str(
                getattr(
                    getattr(issue.fields, "assignee", None),
                    "displayName",
                    "Unassigned",
                )
            ),
            "created": str(getattr(issue.fields, "created", "")),
            "updated": str(getattr(issue.fields, "updated", "")),
            "labels": list(getattr(issue.fields, "labels", []) or []),
            "acceptance_criteria": self._extract_acceptance_criteria(issue),
        }

    def _extract_acceptance_criteria(self, issue) -> str:
        """Extract acceptance criteria from description or custom field"""
        description = getattr(issue.fields, "description", "") or ""
        if not description:
            return ""
        # Simple heuristic: look for an "Acceptance Criteria" header in the description
        lower = description.lower()
        if "acceptance criteria" in lower:
            parts = description.split("Acceptance Criteria")
            if len(parts) > 1:
                return parts[1].strip()
            parts = description.lower().split("acceptance criteria")
            if len(parts) > 1:
                # Re-split the original to preserve casing
                idx = lower.find("acceptance criteria") + len("acceptance criteria")
                return description[idx:].strip()
        return ""

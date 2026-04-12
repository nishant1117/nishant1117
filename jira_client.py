"""
Jira client for fetching epic and ticket data.

Wraps the `jira` SDK and exposes everything the RAG agent needs:
epic details, epic issues, subtasks, linked issues, comments,
changelog, and — crucially — file attachments (with PDF and text
extraction). All methods return plain dicts so they can be serialised
into the vector store or handed to Claude as tool context.

A `search_jql()` method is also provided so the Chat tab's Claude
tool-use loop can construct arbitrary JQL queries on the fly.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional

from jira import JIRA

from config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attachment text-extraction helpers
# ---------------------------------------------------------------------------
_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_EXACT = {
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/yaml",
    "application/javascript",
    "application/x-sh",
    "application/x-python",
    "application/csv",
    "application/x-csv",
}
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".log", ".json", ".yml", ".yaml",
    ".csv", ".tsv", ".xml", ".html", ".htm", ".js", ".ts", ".py",
    ".java", ".go", ".rs", ".rb", ".sh", ".cfg", ".ini", ".toml",
    ".sql", ".graphql", ".proto", ".tf", ".hcl", ".env",
}
_PDF_MIMES = {"application/pdf"}
_PDF_EXTENSIONS = {".pdf"}

# Max bytes downloaded per attachment. 200 KB covers most specs/docs.
_MAX_ATTACHMENT_BYTES = int(os.getenv("MAX_ATTACHMENT_BYTES", "200000"))

# Try importing pypdf for PDF extraction. If unavailable, PDFs degrade
# to metadata-only (filename + page count).
try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    PdfReader = None  # type: ignore[assignment,misc]
    _HAS_PYPDF = False
    logger.info(
        "pypdf not installed — PDF attachment text will not be extracted. "
        "Install with `pip install pypdf` to enable."
    )


def _is_text_like(mime_type: str, filename: str) -> bool:
    mime_type = (mime_type or "").lower()
    if any(mime_type.startswith(p) for p in _TEXT_MIME_PREFIXES):
        return True
    if mime_type in _TEXT_MIME_EXACT:
        return True
    lower = (filename or "").lower()
    return any(lower.endswith(ext) for ext in _TEXT_EXTENSIONS)


def _is_pdf(mime_type: str, filename: str) -> bool:
    if (mime_type or "").lower() in _PDF_MIMES:
        return True
    return any((filename or "").lower().endswith(ext) for ext in _PDF_EXTENSIONS)


def _extract_pdf_text(raw_bytes: bytes, max_chars: int = _MAX_ATTACHMENT_BYTES) -> str:
    """Best-effort text extraction from a PDF blob."""
    if not _HAS_PYPDF:
        return ""
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages: List[str] = []
        total_chars = 0
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            pages.append(text)
            total_chars += len(text)
            if total_chars > max_chars:
                pages.append("...[remaining pages truncated]")
                break
        return "\n\n--- page break ---\n\n".join(pages)
    except Exception as e:
        logger.warning("PDF text extraction failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Jira client
# ---------------------------------------------------------------------------
class JiraClient:
    def __init__(self):
        """Initialize Jira client"""
        self.client = JIRA(
            server=config.JIRA_HOST,
            basic_auth=(config.JIRA_EMAIL, config.JIRA_API_TOKEN),
        )
        logger.info("Connected to Jira: %s", config.JIRA_HOST)

    # ------------------------------------------------------------------
    # JQL search — used by the Chat tab's jira_search tool
    # ------------------------------------------------------------------
    def search_jql(
        self,
        jql: str,
        max_results: int = 50,
        fields: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute an arbitrary JQL query and return a list of issue dicts.

        This is the gateway for Claude to dynamically search Jira. The
        agent_cli `jira_search` tool hands the JQL string it constructs
        straight to this method.
        """
        try:
            fields_str = fields or (
                "summary,description,status,priority,issuetype,"
                "assignee,created,updated,labels,comment,attachment"
            )
            issues = self.client.search_issues(
                jql, maxResults=max_results, fields=fields_str
            )
            results = []
            for issue in issues:
                data = self._extract_issue_data(issue)
                # Inline comment bodies so Claude has them without an
                # extra round-trip.
                comments = []
                try:
                    for c in (issue.fields.comment.comments or []):
                        comments.append({
                            "author": str(getattr(c.author, "displayName", "")),
                            "created": str(getattr(c, "created", "")),
                            "body": getattr(c, "body", "") or "",
                        })
                except Exception:
                    pass
                data["comments"] = comments

                # Inline attachment metadata.
                attachments = []
                try:
                    for att in (getattr(issue.fields, "attachment", None) or []):
                        attachments.append({
                            "filename": getattr(att, "filename", ""),
                            "size": int(getattr(att, "size", 0) or 0),
                            "mime_type": str(getattr(att, "mimeType", "")),
                        })
                except Exception:
                    pass
                data["attachments_meta"] = attachments

                results.append(data)

            logger.info("JQL search returned %d issues", len(results))
            return results
        except Exception as e:
            logger.error("JQL search failed: %s", e)
            return [{"error": str(e)}]

    # ------------------------------------------------------------------
    # Epic-level fetching
    # ------------------------------------------------------------------
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
                    getattr(
                        getattr(issue.fields, "priority", None), "name", "None"
                    )
                ),
                "created": str(getattr(issue.fields, "created", "")),
                "updated": str(getattr(issue.fields, "updated", "")),
            }
        except Exception as e:
            logger.error("Failed to fetch epic %s: %s", epic_key, e)
            return {}

    def get_epic_issues(self, epic_key: str) -> List[Dict[str, Any]]:
        """Fetch all issues linked to an epic.

        Tries multiple JQL strategies for compatibility:
          1. "Epic Link" = KEY   (classic Jira projects)
          2. parent = KEY         (next-gen / team-managed projects)
          3. issuekey in linkedIssues(KEY)
        """
        try:
            jql = (
                f'"Epic Link" = {epic_key} '
                f"OR parent = {epic_key} "
                f"OR key = {epic_key}"
            )
            issues = self.client.search_issues(jql, maxResults=500)
            tickets = []
            for issue in issues:
                if issue.key == epic_key:
                    continue
                data = self._extract_issue_data(issue)
                data["subtasks"] = self.get_issue_subtasks(issue.key)
                data["linked_issues"] = self.get_linked_issues(issue.key)
                tickets.append(data)
            logger.info("Found %d tickets in epic %s", len(tickets), epic_key)
            return tickets
        except Exception as e:
            logger.error("Failed to fetch issues for epic %s: %s", epic_key, e)
            return []

    # ------------------------------------------------------------------
    # Per-issue context
    # ------------------------------------------------------------------
    def get_issue_subtasks(self, issue_key: str) -> List[Dict[str, Any]]:
        """Fetch all subtasks for an issue — including their full
        descriptions so Claude can reason about child ticket intent."""
        try:
            issue = self.client.issue(issue_key)
            subtasks = []
            if hasattr(issue.fields, "subtasks"):
                for sub in issue.fields.subtasks:
                    # Fetch the full subtask to get its description.
                    try:
                        full_sub = self.client.issue(sub.key)
                        desc = getattr(full_sub.fields, "description", "") or ""
                    except Exception:
                        desc = ""
                    subtasks.append({
                        "key": sub.key,
                        "summary": getattr(sub.fields, "summary", ""),
                        "description": desc,
                        "status": str(
                            getattr(sub.fields.status, "name", "None")
                        ),
                        "priority": str(
                            getattr(
                                getattr(sub.fields, "priority", None),
                                "name",
                                "None",
                            )
                        ),
                        "assignee": str(
                            getattr(
                                getattr(sub.fields, "assignee", None),
                                "displayName",
                                "Unassigned",
                            )
                        ),
                    })
            logger.info("Found %d subtasks for %s", len(subtasks), issue_key)
            return subtasks
        except Exception as e:
            logger.error("Failed to fetch subtasks for %s: %s", issue_key, e)
            return []

    def get_linked_issues(self, issue_key: str) -> List[Dict[str, Any]]:
        """Fetch all linked issues for an issue — with descriptions."""
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

                    # Fetch full issue to get description.
                    desc = ""
                    try:
                        full_other = self.client.issue(other.key)
                        desc = (
                            getattr(full_other.fields, "description", "") or ""
                        )
                    except Exception:
                        pass

                    linked.append({
                        "key": other.key,
                        "summary": getattr(other.fields, "summary", ""),
                        "description": desc,
                        "status": str(
                            getattr(other.fields.status, "name", "None")
                        ),
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
            logger.info(
                "Found %d linked issues for %s", len(linked), issue_key
            )
            return linked
        except Exception as e:
            logger.error(
                "Failed to fetch linked issues for %s: %s", issue_key, e
            )
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
            logger.error("Failed to fetch comments for %s: %s", issue_key, e)
            return []

    def get_issue_changelog(self, issue_key: str) -> List[Dict[str, Any]]:
        """Fetch changelog/history for an issue.

        Returns a flat list of field-level changes with explicit
        `is_status_change` flag so prompts can surface status
        transitions prominently.
        """
        try:
            issue = self.client.issue(issue_key, expand="changelog")
            history = []
            if hasattr(issue, "changelog"):
                for h in issue.changelog.histories:
                    for item in h.items:
                        field = getattr(item, "field", "") or ""
                        history.append({
                            "created": str(getattr(h, "created", "")),
                            "author": str(
                                getattr(h.author, "displayName", "")
                            ),
                            "field": field,
                            "from_value": getattr(item, "fromString", "") or "",
                            "to_value": getattr(item, "toString", "") or "",
                            "is_status_change": field.lower() == "status",
                        })
            return history
        except Exception as e:
            logger.error("Failed to fetch changelog for %s: %s", issue_key, e)
            return []

    # ------------------------------------------------------------------
    # Attachment fetching with PDF + text extraction
    # ------------------------------------------------------------------
    def get_issue_attachments(
        self,
        issue_key: str,
        download_text: bool = True,
        max_bytes: int = _MAX_ATTACHMENT_BYTES,
    ) -> List[Dict[str, Any]]:
        """Fetch attachments for an issue.

        - Text-like files (txt, md, json, csv, yaml, py, etc.) are
          downloaded and decoded as UTF-8.
        - PDF files are downloaded and text-extracted via pypdf (if
          installed). Falls back to metadata-only if pypdf is missing.
        - Binary files (images, Office docs, etc.) are represented by
          metadata only (filename, size, mime_type, author).

        The `max_bytes` cap applies to both text and PDF downloads
        (default 200 KB, overridable via the MAX_ATTACHMENT_BYTES env
        var).
        """
        try:
            issue = self.client.issue(issue_key)
            attachments_raw = getattr(issue.fields, "attachment", None) or []
            out: List[Dict[str, Any]] = []

            for att in attachments_raw:
                mime_type = str(getattr(att, "mimeType", "")) or ""
                filename = getattr(att, "filename", "") or ""
                size = int(getattr(att, "size", 0) or 0)
                url = getattr(att, "content", "") or ""

                record: Dict[str, Any] = {
                    "filename": filename,
                    "size": size,
                    "mime_type": mime_type,
                    "author": str(
                        getattr(
                            getattr(att, "author", None), "displayName", ""
                        )
                    ),
                    "created": str(getattr(att, "created", "")),
                    "url": url,
                    "text_content": None,
                    "extraction_method": None,
                }

                if not download_text or not url or size > max_bytes:
                    out.append(record)
                    continue

                # Try to download the blob.
                raw_bytes: bytes | None = None
                try:
                    session = getattr(self.client, "_session", None)
                    if session is not None:
                        resp = session.get(url)
                        if resp.status_code == 200:
                            raw_bytes = resp.content
                except Exception as e:
                    logger.warning(
                        "Download failed for %s on %s: %s",
                        filename,
                        issue_key,
                        e,
                    )

                if raw_bytes is None:
                    out.append(record)
                    continue

                # Decide extraction strategy.
                if _is_pdf(mime_type, filename):
                    text = _extract_pdf_text(raw_bytes, max_chars=max_bytes)
                    if text:
                        record["text_content"] = text
                        record["extraction_method"] = "pdf_pypdf"
                    else:
                        record["extraction_method"] = "pdf_no_text"
                elif _is_text_like(mime_type, filename):
                    text = raw_bytes.decode("utf-8", errors="replace")
                    record["text_content"] = text[:max_bytes]
                    record["extraction_method"] = "text_utf8"
                else:
                    record["extraction_method"] = "binary_skip"

                out.append(record)

            logger.info(
                "Found %d attachments for %s (%d with text)",
                len(out),
                issue_key,
                sum(1 for r in out if r.get("text_content")),
            )
            return out
        except Exception as e:
            logger.error(
                "Failed to fetch attachments for %s: %s", issue_key, e
            )
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_issue_data(self, issue) -> Dict[str, Any]:
        """Extract relevant data from a Jira issue"""
        return {
            "key": issue.key,
            "summary": getattr(issue.fields, "summary", "") or "",
            "description": getattr(issue.fields, "description", "") or "",
            "status": str(getattr(issue.fields.status, "name", "None")),
            "priority": str(
                getattr(
                    getattr(issue.fields, "priority", None), "name", "None"
                )
            ),
            "issue_type": str(
                getattr(
                    getattr(issue.fields, "issuetype", None), "name", "None"
                )
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
        lower = description.lower()
        if "acceptance criteria" in lower:
            idx = lower.find("acceptance criteria") + len("acceptance criteria")
            return description[idx:].strip()
        return ""

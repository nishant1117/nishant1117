"""
FastAPI application exposing the Jira RAG agent.

Endpoints:
  POST /process-epic            async processing with background task
  POST /process-epic-sync       sync processing; returns results inline
  GET  /results/{epic_key}      fetch cached results
  GET  /generate-report/{epic}  plain-text report
  GET  /analysis/{epic_key}     summarised analysis
  GET  /edge-cases/{epic_key}
  GET  /regression/{epic_key}
  GET  /bug-analysis/{epic_key}
  POST /compare-epics           delta analysis between two epics
  POST /custom-analysis         free-form user prompt against an epic
  GET  /insights/{epic_key}
  GET  /recommendations/{epic_key}
  GET  /action-plan/{epic_key}
  POST /impact-analysis
"""
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from rag_agent import JiraRAGAgent
from test_generator import TestCaseGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jira RAG Agent",
    description="RAG agent for Jira Epic analysis and test case generation",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache of processed epics. Swap for Redis in production.
results_cache: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------
class EpicProcessRequest(BaseModel):
    epic_key: str
    description: Optional[str] = None


class ComparisonRequest(BaseModel):
    epic_key_1: str
    epic_key_2: str


class CustomAnalysisRequest(BaseModel):
    epic_key: str
    custom_prompt: str


class ImpactAnalysisRequest(BaseModel):
    epic_key: str
    changes: List[str]


class ProcessingResponse(BaseModel):
    status: str
    epic_key: str
    message: str


# ---------------------------------------------------------------------------
# Lifecycle + health
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Jira RAG Agent API")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Jira RAG Agent"}


# ---------------------------------------------------------------------------
# Epic processing
# ---------------------------------------------------------------------------
@app.post("/process-epic", response_model=ProcessingResponse)
async def process_epic(
    request: EpicProcessRequest, background_tasks: BackgroundTasks
):
    """
    Process a Jira Epic and generate test cases and analysis.

    Kicks off a background task and returns immediately. Poll
    /results/{epic_key} for the output.
    """
    try:
        logger.info(f"Received request to process epic: {request.epic_key}")
        background_tasks.add_task(_process_epic_task, request.epic_key)
        return ProcessingResponse(
            status="processing",
            epic_key=request.epic_key,
            message=(
                f"Epic {request.epic_key} is being processed. "
                "Results will be available shortly."
            ),
        )
    except Exception as e:
        logger.error(f"Error processing epic request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-epic-sync")
async def process_epic_sync(request: EpicProcessRequest):
    """
    Process a Jira Epic synchronously and return results directly.
    Warning: This may take a long time for large epics.
    """
    try:
        logger.info(f"Starting synchronous processing of epic: {request.epic_key}")
        agent = JiraRAGAgent()
        results = agent.process_epic(request.epic_key)
        results_cache[request.epic_key] = results
        logger.info(f"Completed processing epic: {request.epic_key}")
        return {"status": "completed", "epic_key": request.epic_key, "results": results}
    except Exception as e:
        logger.error(f"Error processing epic: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{epic_key}")
async def get_results(epic_key: str):
    """Get results for a processed epic"""
    if epic_key not in results_cache:
        raise HTTPException(
            status_code=404, detail=f"No results found for epic {epic_key}"
        )
    return results_cache[epic_key]


@app.get("/generate-report/{epic_key}")
async def generate_report(epic_key: str):
    """Generate a formatted report for a processed epic"""
    if epic_key not in results_cache:
        raise HTTPException(
            status_code=404, detail=f"No results found for epic {epic_key}"
        )
    try:
        agent = JiraRAGAgent()
        return {"report": agent.generate_report(results_cache[epic_key])}
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Analyzer endpoints
# ---------------------------------------------------------------------------
@app.get("/analysis/{epic_key}")
async def get_analysis(epic_key: str):
    """Get comprehensive analysis including test cases, edge cases, regression."""
    if epic_key not in results_cache:
        raise HTTPException(
            status_code=404, detail=f"No results found for epic {epic_key}"
        )
    try:
        results = results_cache[epic_key]
        epic_details = results.get("epic_details", {}) or {}
        tickets = results.get("tickets", {}) or {}
        ticket_details = {}
        for k, t in tickets.items():
            ticket_details[k] = {
                "summary": t.get("ticket", {}).get("summary", "N/A"),
                "status": t.get("ticket", {}).get("status", "N/A"),
                "priority": t.get("ticket", {}).get("priority", "N/A"),
                "test_cases": len(
                    t.get("test_cases", {}).get("test_cases", [])
                ),
                "edge_cases": len(
                    t.get("edge_cases", {}).get("edge_cases", [])
                ),
                "analysis": t.get("analysis", {}),
                "regression_analysis": t.get("regression_analysis", {}),
            }
        return {
            "epic_key": epic_key,
            "epic_details": epic_details,
            "ticket_details": ticket_details,
        }
    except Exception as e:
        logger.error(f"Error retrieving analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/edge-cases/{epic_key}")
async def get_edge_cases(epic_key: str):
    """Get all edge cases for an epic"""
    if epic_key not in results_cache:
        raise HTTPException(
            status_code=404, detail=f"No results found for epic {epic_key}"
        )
    try:
        by_ticket = {}
        total = 0
        for k, t in (results_cache[epic_key].get("tickets") or {}).items():
            edge_cases = t.get("edge_cases", {}).get("edge_cases", [])
            by_ticket[k] = edge_cases
            total += len(edge_cases)
        return {"by_ticket": by_ticket, "total_edge_cases": total}
    except Exception as e:
        logger.error(f"Error retrieving edge cases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/regression/{epic_key}")
async def get_regression_analysis(epic_key: str):
    """Get regression analysis for an epic"""
    if epic_key not in results_cache:
        raise HTTPException(
            status_code=404, detail=f"No results found for epic {epic_key}"
        )
    try:
        by_ticket = {}
        risk_buckets = {"High": [], "Medium": [], "Low": []}
        for k, t in (results_cache[epic_key].get("tickets") or {}).items():
            reg = t.get("regression_analysis", {}) or {}
            by_ticket[k] = reg
            level = reg.get("regression_risk_level", "Low")
            risk_buckets.setdefault(level, []).append(k)
        return {
            "by_ticket": by_ticket,
            "overall_risk_assessment": {
                "high_risk_tickets": risk_buckets.get("High", []),
                "medium_risk_tickets": risk_buckets.get("Medium", []),
                "low_risk_tickets": risk_buckets.get("Low", []),
            },
        }
    except Exception as e:
        logger.error(f"Error retrieving regression analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bug-analysis/{epic_key}")
async def get_bug_analysis(epic_key: str):
    """Get bug detection analysis for an epic"""
    if epic_key not in results_cache:
        raise HTTPException(
            status_code=404, detail=f"No results found for epic {epic_key}"
        )
    try:
        by_ticket = {}
        all_critical: List[Dict[str, Any]] = []
        all_vulns: List[Dict[str, Any]] = []
        all_dep_risks: List[Dict[str, Any]] = []
        high_risk_areas = set()

        for k, t in (results_cache[epic_key].get("tickets") or {}).items():
            ba = t.get("bug_analysis", {}) or {}
            by_ticket[k] = ba
            for b in ba.get("critical_bugs", []) or []:
                all_critical.append({"ticket": k, **b})
            for v in ba.get("potential_vulnerabilities", []) or []:
                all_vulns.append({"ticket": k, **v})
            for d in ba.get("dependency_risks", []) or []:
                all_dep_risks.append({"ticket": k, **d})
            high_risk_areas.update(ba.get("high_risk_areas", []) or [])

        return {
            "by_ticket": by_ticket,
            "total_critical_bugs": len(all_critical),
            "all_critical_bugs": all_critical,
            "total_vulnerabilities": len(all_vulns),
            "all_vulnerabilities": all_vulns,
            "total_dependency_risks": len(all_dep_risks),
            "all_dependency_risks": all_dep_risks,
            "high_risk_areas": list(high_risk_areas),
        }
    except Exception as e:
        logger.error(f"Error retrieving bug analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Comparator endpoint
# ---------------------------------------------------------------------------
@app.post("/compare-epics")
async def compare_epics(request: ComparisonRequest):
    """Compare two epics and generate delta analysis"""
    try:
        logger.info(
            f"Comparing epics {request.epic_key_1} and {request.epic_key_2}"
        )
        agent = JiraRAGAgent()

        for key in (request.epic_key_1, request.epic_key_2):
            if key not in results_cache:
                logger.info(f"Processing {key} for comparison")
                results_cache[key] = agent.process_epic(key)

        tg = TestCaseGenerator()
        context1 = agent.vector_store.search(
            f"epic {request.epic_key_1}"
        )
        context2 = agent.vector_store.search(
            f"epic {request.epic_key_2}"
        )
        result = tg.compare_epics(
            results_cache[request.epic_key_1].get("epic_details", {}),
            results_cache[request.epic_key_2].get("epic_details", {}),
            context1,
            context2,
        )
        return {"success": True, "comparison": result}
    except Exception as e:
        logger.error(f"Error comparing epics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Free-form prompts
# ---------------------------------------------------------------------------
@app.post("/custom-analysis")
async def custom_analysis(request: CustomAnalysisRequest):
    """Generate custom analysis based on user prompt"""
    try:
        logger.info(f"Generating custom analysis for {request.epic_key}")
        if request.epic_key not in results_cache:
            agent = JiraRAGAgent()
            results_cache[request.epic_key] = agent.process_epic(request.epic_key)
        epic_results = results_cache[request.epic_key]

        # Flatten all ticket data into a single combined call
        combined_ticket = epic_results.get("epic_details", {})
        all_comments: List[Dict[str, Any]] = []
        all_changelog: List[Dict[str, Any]] = []
        for t in (epic_results.get("tickets") or {}).values():
            all_comments.extend(t.get("comments", []) or [])
            all_changelog.extend(t.get("changelog", []) or [])

        agent = JiraRAGAgent()
        context = agent.vector_store.search(request.custom_prompt)

        tg = TestCaseGenerator()
        result = tg.custom_prompt_analysis(
            combined_ticket, all_comments, all_changelog, context,
            request.custom_prompt,
        )
        return {"success": True, "analysis": result}
    except Exception as e:
        logger.error(f"Error generating custom analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insights/{epic_key}")
async def get_insights(epic_key: str):
    """Get Claude AI insights for an epic"""
    try:
        logger.info(f"Generating insights for {epic_key}")
        if epic_key not in results_cache:
            agent = JiraRAGAgent()
            results_cache[epic_key] = agent.process_epic(epic_key)
        tg = TestCaseGenerator()
        result = tg.generate_data_insights(
            results_cache[epic_key].get("epic_details", {}),
            results_cache[epic_key],
        )
        return {"success": True, "insights": result}
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/{epic_key}")
async def get_recommendations(epic_key: str):
    """Get Claude AI recommendations for addressing issues"""
    try:
        logger.info(f"Generating recommendations for {epic_key}")
        if epic_key not in results_cache:
            agent = JiraRAGAgent()
            results_cache[epic_key] = agent.process_epic(epic_key)

        issues: List[Dict[str, Any]] = []
        for t in (results_cache[epic_key].get("tickets") or {}).values():
            for b in t.get("bug_analysis", {}).get("critical_bugs", []) or []:
                issues.append({
                    "type": "Critical Bug",
                    "description": b.get("title", "Unknown bug"),
                })
            if t.get("regression_analysis", {}).get("regression_risk_level") in (
                "High", "Medium"
            ):
                issues.append({
                    "type": "Regression Risk",
                    "description": t.get("ticket", {}).get("summary", ""),
                })

        tg = TestCaseGenerator()
        result = tg.generate_recommendations(
            results_cache[epic_key].get("epic_details", {}), issues
        )
        return {"success": True, "recommendations": result, "issue_count": len(issues)}
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/action-plan/{epic_key}")
async def get_action_plan(epic_key: str):
    """Get Claude AI multi-action orchestration plan"""
    try:
        logger.info(f"Generating action plan for {epic_key}")
        if epic_key not in results_cache:
            agent = JiraRAGAgent()
            results_cache[epic_key] = agent.process_epic(epic_key)
        tg = TestCaseGenerator()
        result = tg.generate_multi_action_plan(
            results_cache[epic_key].get("epic_details", {}),
            results_cache[epic_key],
        )
        return {"success": True, "action_plan": result}
    except Exception as e:
        logger.error(f"Error generating action plan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/impact-analysis")
async def get_impact_analysis(request: ImpactAnalysisRequest):
    """Get Claude AI impact analysis for changes"""
    try:
        logger.info(f"Analyzing impact for {request.epic_key}")
        if request.epic_key not in results_cache:
            agent = JiraRAGAgent()
            results_cache[request.epic_key] = agent.process_epic(request.epic_key)
        tg = TestCaseGenerator()
        result = tg.generate_impact_analysis(
            results_cache[request.epic_key].get("epic_details", {}),
            request.changes,
        )
        return {"success": True, "impact_analysis": result}
    except Exception as e:
        logger.error(f"Error analyzing impact: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------
def _process_epic_task(epic_key: str):
    """Background task to process epic"""
    try:
        logger.info(f"Background task: Processing epic {epic_key}")
        agent = JiraRAGAgent()
        results_cache[epic_key] = agent.process_epic(epic_key)
        logger.info(f"Background task completed for epic {epic_key}")
    except Exception as e:
        logger.error(f"Background task failed for epic {epic_key}: {str(e)}")
        results_cache[epic_key] = {"status": "failed", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

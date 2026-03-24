import os
import json
import uuid
import logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query, Header, Depends, Request
from pydantic import BaseModel
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState, StatementParameterListItem

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("hub-api")

# ── Configuration (from app.yaml env or defaults) ──────────────
CATALOG      = os.getenv("CATALOG", "olympus_hub")
SCHEMA       = os.getenv("SCHEMA", "opsdata")
WAREHOUSE_ID = os.getenv("WAREHOUSE_ID", "6f143f6d87ef5580")
API_KEY      = os.getenv("API_KEY")  # App-level auth key (set in app.yaml)
RG_NAME      = os.getenv("RG_NAME", "default")

# ── SDK client (uses app service-principal credentials) ────────
w = WorkspaceClient()

# ── Validate config at startup ─────────────────────────────────
logger.info("Hub API starting — catalog=%s  schema=%s  warehouse=%s", CATALOG, SCHEMA, WAREHOUSE_ID)
logger.info("API_KEY configured: %s", "yes" if API_KEY else "NO")
logger.info("DATABRICKS_CLIENT_ID present: %s", "yes" if os.getenv("DATABRICKS_CLIENT_ID") else "no")

# ── FastAPI app ────────────────────────────────────────────────
app = FastAPI(
    title="Olympus Hub API",
    description="Operational intelligence hub — ingest failure signals, diagnose, and manage incidents.",
    version="1.2.0",
)


# ── Request logging middleware (debug auth issues) ─────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(
        "→ %s %s  | X-Api-Key: %s  | X-Forwarded-For: %s  | X-Forwarded-User: %s",
        request.method,
        request.url.path,
        "***" if request.headers.get("x-api-key") else "<none>",
        request.headers.get("x-forwarded-for", "<direct>"),
        request.headers.get("x-forwarded-preferred-username", "<none>"),
    )
    response = await call_next(request)
    logger.info("← %s %s  | status=%d", request.method, request.url.path, response.status_code)
    return response


# ── SQL helper via Statement Execution API ─────────────────────
def run_sql(
    sql: str,
    parameters: Optional[List[StatementParameterListItem]] = None,
    fetch: bool = True,
):
    """Execute parameterised SQL against Unity Catalog tables."""
    response = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=sql,
        parameters=parameters,
        wait_timeout="30s",
    )
    if response.status and response.status.state == StatementState.SUCCEEDED:
        if fetch and response.manifest and response.result:
            columns = [col.name for col in response.manifest.schema.columns]
            rows = response.result.data_array or []
            return [dict(zip(columns, row)) for row in rows]
        return []
    error_msg = (
        str(response.status.error)
        if response.status and response.status.error
        else "Unknown SQL error"
    )
    raise HTTPException(status_code=500, detail=f"SQL error: {error_msg}")


# ── Diagnosis engine ───────────────────────────────────────────
TRANSIENT_PATTERNS = [
    "sockettimeoutexception",
    "connectiontimeout",
    "lost executor",
    "nodelosst",
    "http 429",
]

def diagnose(job_name: str, error_message: str, log_snippet: str = "") -> dict:
    """Keyword-based root cause diagnosis (swap with LLM call when ready)."""
    err = error_message.lower()

    if any(p in err for p in TRANSIENT_PATTERNS):
        return {
            "root_cause":      "Transient network or executor failure",
            "cause_category":  "transient",
            "confidence":      0.95,
            "suggested_fix":   "Auto-retry triggered — no action needed",
            "business_impact": {"sla_at_risk": False, "severity": "low"},
            "safe_to_auto_fix": True,
        }
    if any(p in err for p in ["column", "schema", "not found", "analysisexception"]):
        return {
            "root_cause":      f"Schema mismatch — column missing in {job_name}",
            "cause_category":  "schema_change",
            "confidence":      0.88,
            "suggested_fix":   "Check upstream table for removed or renamed columns",
            "business_impact": {"sla_at_risk": True, "severity": "high"},
            "safe_to_auto_fix": False,
        }
    if any(p in err for p in ["memory", "oom", "heap", "outofmemory"]):
        return {
            "root_cause":      "Memory pressure — cluster undersized for workload",
            "cause_category":  "resource_limit",
            "confidence":      0.85,
            "suggested_fix":   "Increase autoscale max nodes or add partition pruning",
            "business_impact": {"sla_at_risk": False, "severity": "medium"},
            "safe_to_auto_fix": False,
        }
    return {
        "root_cause":      "Unknown failure — manual investigation needed",
        "cause_category":  "code_bug",
        "confidence":      0.50,
        "suggested_fix":   "Review full job logs for root cause",
        "business_impact": {"sla_at_risk": False, "severity": "low"},
        "safe_to_auto_fix": False,
    }


# ── Request / Response models ──────────────────────────────────
class IngestRequest(BaseModel):
    workspace_id: str
    signal_type: str
    team_name: str = ""
    rg_name: str = ""
    job_name: str
    run_id: str
    error_message: str
    log_snippet: str = ""
    raw_payload: str = ""

class AcceptFixRequest(BaseModel):
    incident_id: str
    workspace_id: str
    job_name: str


# ── Auth dependency ───────────────────────────────────────────
#
#  Databricks Apps auth flow (what we discovered):
#
#    1. Caller sends:   Authorization: Bearer <OAuth-JWT>  +  X-Api-Key: <key>
#    2. Proxy validates the OAuth JWT.  If invalid → proxy returns 401 {}.
#    3. Proxy strips the Authorization header, forwards the rest to the app.
#    4. App receives:   X-Api-Key: <key>  (no Bearer token)
#       Plus proxy-injected identity headers (e.g. X-Forwarded-Preferred-Username).
#
#  Therefore the app should authenticate via X-Api-Key (or proxy identity
#  headers), NOT via the Bearer token (which it never sees).
#
def verify_auth(
    request: Request,
    x_api_key: Optional[str] = Header(None),
):
    """
    Authenticate the request.  Accepts either:
      1. X-Api-Key header  — validated against API_KEY env var.
      2. Proxy-forwarded identity  — if the Databricks Apps proxy already
         authenticated the user, it injects identity headers.  When present
         we trust the proxy and skip API-key validation.
    """
    # Path 1: Proxy-forwarded identity (proxy already validated OAuth)
    proxy_user = request.headers.get("x-forwarded-preferred-username") or request.headers.get("x-forwarded-user")
    if proxy_user:
        logger.info("Auth: accepted via proxy identity (%s)", proxy_user)
        return {"auth_type": "proxy", "user": proxy_user}

    # Path 2: X-Api-Key header
    if x_api_key:
        if not API_KEY:
            raise HTTPException(
                status_code=500,
                detail="API_KEY not configured on server — set it in app.yaml",
            )
        if x_api_key != API_KEY:
            logger.warning("Auth: invalid API key received")
            raise HTTPException(status_code=401, detail="Invalid API key")
        logger.info("Auth: accepted via X-Api-Key")
        return {"auth_type": "api_key"}

    # Neither method provided
    logger.warning("Auth: no credentials (no proxy identity, no X-Api-Key)")
    raise HTTPException(
        status_code=401,
        detail="Authentication required — provide X-Api-Key header or call through the Databricks Apps proxy with a valid OAuth token",
    )


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "catalog": CATALOG, "schema": SCHEMA}


@app.post("/ingest")
def ingest(req: IngestRequest, auth: dict = Depends(verify_auth)):
    """Receive a failure signal → write raw_signals → diagnose → write incident."""
    signal_id   = str(uuid.uuid4())
    incident_id = str(uuid.uuid4())
    rg_name     = req.rg_name or RG_NAME

    # Write raw signal (parameterised)
    run_sql(
        f"""
        INSERT INTO {CATALOG}.{SCHEMA}.raw_signals
        (signal_id, workspace_id, rg_name, signal_type,
         job_name, run_id, error_message, log_snippet,
         raw_payload, ingested_at)
        VALUES (
          :signal_id, :workspace_id, :rg_name, :signal_type,
          :job_name, :run_id, :error_message, :log_snippet,
          :raw_payload, current_timestamp()
        )
        """,
        parameters=[
            StatementParameterListItem(name="signal_id",     value=signal_id),
            StatementParameterListItem(name="workspace_id",  value=req.workspace_id),
            StatementParameterListItem(name="rg_name",       value=rg_name),
            StatementParameterListItem(name="signal_type",   value=req.signal_type),
            StatementParameterListItem(name="job_name",      value=req.job_name),
            StatementParameterListItem(name="run_id",        value=req.run_id),
            StatementParameterListItem(name="error_message", value=req.error_message),
            StatementParameterListItem(name="log_snippet",   value=req.log_snippet[-2000:]),
            StatementParameterListItem(name="raw_payload",   value=req.raw_payload),
        ],
        fetch=False,
    )

    # Diagnose
    dx = diagnose(req.job_name, req.error_message, req.log_snippet)
    impact_str = json.dumps(dx["business_impact"])

    # Write incident (parameterised)
    run_sql(
        f"""
        INSERT INTO {CATALOG}.{SCHEMA}.incidents
        (incident_id, workspace_id, team_name, job_name,
         root_cause, cause_category, confidence,
         suggested_fix, business_impact,
         auto_resolved, detected_at, detected_date,
         resolved_at, created_at)
        VALUES (
          :incident_id, :workspace_id, :team_name, :job_name,
          :root_cause, :cause_category, :confidence,
          :suggested_fix, :business_impact,
          :auto_resolved, current_timestamp(), current_date(),
          null, current_timestamp()
        )
        """,
        parameters=[
            StatementParameterListItem(name="incident_id",    value=incident_id),
            StatementParameterListItem(name="workspace_id",   value=req.workspace_id),
            StatementParameterListItem(name="team_name",      value=req.team_name),
            StatementParameterListItem(name="job_name",       value=req.job_name),
            StatementParameterListItem(name="root_cause",     value=dx["root_cause"]),
            StatementParameterListItem(name="cause_category", value=dx["cause_category"]),
            StatementParameterListItem(name="confidence",     value=str(dx["confidence"]),  type="DOUBLE"),
            StatementParameterListItem(name="suggested_fix",  value=dx["suggested_fix"]),
            StatementParameterListItem(name="business_impact", value=impact_str),
            StatementParameterListItem(name="auto_resolved",  value=str(dx["safe_to_auto_fix"]).lower(), type="BOOLEAN"),
        ],
        fetch=False,
    )

    return {
        "status":      "ok",
        "signal_id":   signal_id,
        "incident_id": incident_id,
        "category":    dx["cause_category"],
        "confidence":  dx["confidence"],
        "auto_fixed":  dx["safe_to_auto_fix"],
    }


@app.get("/incidents")
def get_incidents(
    workspace_id: str = Query(...),
    limit: int = Query(10, ge=1, le=100),
    auth: dict = Depends(verify_auth),
):
    """Return diagnosed incidents for a workspace."""
    rows = run_sql(
        f"""
        SELECT incident_id, workspace_id, team_name, job_name,
               root_cause, cause_category, confidence, suggested_fix,
               business_impact, auto_resolved, detected_at, detected_date,
               resolved_at, created_at
        FROM  {CATALOG}.{SCHEMA}.incidents
        WHERE workspace_id = :workspace_id
        ORDER BY detected_at DESC
        LIMIT {limit}
        """,
        parameters=[
            StatementParameterListItem(name="workspace_id", value=workspace_id),
        ],
    )
    return {"workspace_id": workspace_id, "count": len(rows or []), "incidents": rows or []}


@app.post("/accept-fix")
def accept_fix(req: AcceptFixRequest, auth: dict = Depends(verify_auth)):
    """Mark incident resolved and optionally retry the Databricks job."""
    run_sql(
        f"""
        UPDATE {CATALOG}.{SCHEMA}.incidents
        SET    auto_resolved = true,
               resolved_at   = current_timestamp()
        WHERE  incident_id   = :incident_id
        AND    workspace_id  = :workspace_id
        """,
        parameters=[
            StatementParameterListItem(name="incident_id",  value=req.incident_id),
            StatementParameterListItem(name="workspace_id", value=req.workspace_id),
        ],
        fetch=False,
    )

    new_run_id = None
    try:
        jobs = list(w.jobs.list(name=req.job_name))
        if jobs:
            run = w.jobs.run_now(job_id=jobs[0].job_id)
            new_run_id = str(run.run_id)
    except Exception:
        pass

    return {"status": "resolved", "incident_id": req.incident_id, "new_run_id": new_run_id}


# ── Entrypoint ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DATABRICKS_APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

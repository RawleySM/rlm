# User Story: Dual-Mode Execution for RLM-ADK (Local + Native Databricks)

## Overview

This document defines the requirements for enabling `rlm_adk` to support **dual-mode execution**: both **Local Orchestration** (running on developer workstations via REST APIs) and **Native Execution** (running directly on Databricks clusters as a scheduled Job). The architecture maintains both modes with automatic detection and explicit override capability.

---

## User Story

**As a** Healthcare Data Engineer responsible for vendor master data management,

**I want** the RLM-ADK agent to support both local development and native Databricks execution,

**So that** I can:
1. Develop and test locally with fast iteration cycles using REST APIs.
2. Deploy the same codebase to Databricks clusters for production workloads.
3. Process full-scale ERP datasets (millions of vendor records) without local hardware limitations.
4. Keep sensitive healthcare data within the Databricks security perimeter.
5. Schedule automated vendor resolution pipelines as recurring Databricks Jobs.
6. Leverage Spark's distributed compute for heavy data transformations before LLM reasoning.

---

## Benefits

### 1. Data Gravity (Performance)
Moving large datasets (1GB+ of vendor records) from Delta Lake to the Driver's memory is nearly instantaneous on Databricks. The current architecture requires downloading data over REST APIs to a local machine, introducing significant network latency and potential timeouts. Native execution eliminates this bottleneck.

### 2. Infinite Scale (Distributed Processing)
When running natively, the `rlm_execute_code` tool can spawn **Spark Jobs**. Instead of the LLM only "reasoning" about data in a local Python list, it can instruct Spark to perform massive joins across billions of rows, then pull only the "top N discrepancies" into its reasoning context. This enables true large-scale vendor resolution.

### 3. Security & Compliance
Sensitive healthcare data **never leaves the Databricks security perimeter**. Only prompts and reasoning steps are sent to Gemini. The local developer workstation never sees raw PII (Personally Identifiable Information) from vendor records. This is critical for HIPAA compliance and enterprise security policies.

### 4. Productionization
The `root_agent` can be scheduled as a **Databricks Workflow Job**. For example, the "Vendor Resolution Pipeline" can run automatically every Monday at 2 AM without requiring a local machine to be online. This enables hands-off, production-grade automation.

---

## Architecture: Local Mode (Retained)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Developer Workstation                                              │
│  ┌───────────────────┐                                              │
│  │   rlm_adk Agent   │                                              │
│  │   (root_agent)    │                                              │
│  └─────────┬─────────┘                                              │
│            │                                                        │
│  ┌─────────▼─────────┐     REST API      ┌────────────────────────┐ │
│  │  databricks_repl  │◄─────────────────►│   Databricks Workspace │ │
│  │  unity_catalog    │  (requests lib)   │   - Unity Catalog      │ │
│  └─────────┬─────────┘                   │   - SQL Warehouse      │ │
│            │                             │   - Clusters           │ │
│  ┌─────────▼─────────┐                   └────────────────────────┘ │
│  │   Local REPL      │                                              │
│  │   (rlm_repl.py)   │  context = [...]  (Python list in RAM)      │
│  └─────────┬─────────┘                                              │
│            │                                                        │
│  ┌─────────▼─────────┐     HTTPS         ┌────────────────────────┐ │
│  │   llm_bridge.py   │◄─────────────────►│   Gemini API           │ │
│  └───────────────────┘                   └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

**Use Cases**:
- Local development and testing with fast iteration.
- Small-to-medium datasets that fit in local memory.
- Debugging agent behavior before production deployment.

**Trade-offs**:
- Data must be downloaded to local machine (bandwidth/memory constraints).
- Requires `DATABRICKS_TOKEN` stored on local machine.
- Cannot leverage Spark for distributed processing.
- Cannot run unattended as a scheduled job.

---

## Architecture: Native Mode (New)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Databricks Workspace                                               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Databricks Job (Cluster: ML Runtime)                         │  │
│  │  ┌───────────────────┐                                        │  │
│  │  │   rlm_adk Agent   │  (Installed as cluster library)        │  │
│  │  │   (root_agent)    │                                        │  │
│  │  └─────────┬─────────┘                                        │  │
│  │            │                                                  │  │
│  │  ┌─────────▼─────────┐                                        │  │
│  │  │  databricks_repl  │  spark.sql(query)  (Direct SparkSession│  │
│  │  │  unity_catalog    │  dbutils, etc.      access)            │  │
│  │  └─────────┬─────────┘                                        │  │
│  │            │                                                  │  │
│  │  ┌─────────▼─────────┐                                        │  │
│  │  │   Native REPL     │  context = spark.table("...")          │  │
│  │  │   (rlm_repl.py)   │  (Spark DataFrame or collected list)   │  │
│  │  └─────────┬─────────┘                                        │  │
│  │            │                                                  │  │
│  │  ┌─────────▼─────────┐     HTTPS         ┌──────────────────┐ │  │
│  │  │   llm_bridge.py   │◄─────────────────►│   Gemini API     │ │  │
│  │  └───────────────────┘                   └──────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Unity Catalog ◄──► Delta Tables ◄──► Volumes                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Use Cases**:
- Production workloads with large datasets (millions of records).
- Scheduled automated pipelines (Databricks Workflow Jobs).
- Security-sensitive environments (data never leaves Databricks perimeter).

**Advantages**:
- Data stays within Databricks (no network transfer to local machine).
- Uses **Ambient Credentials** (no tokens stored locally).
- Can leverage `spark` for distributed joins/aggregations.
- Runs as a scheduled Databricks Workflow Job.

---

## Mode Detection & Configuration

The architecture uses a **single source of truth** for mode detection via `rlm_adk/runtime.py`:

### Environment Variable Schema

| Variable | Purpose | Values |
|----------|---------|--------|
| `RLM_EXECUTION_MODE` | Force mode override | `local`, `native`, or unset (auto-detect) |
| `DATABRICKS_RUNTIME_VERSION` | Auto-detection marker | Set by Databricks runtime |
| `RLM_LLM_TIMEOUT_SECONDS` | LLM call timeout | Integer (default: 60) |
| `RLM_LLM_MAX_RETRIES` | LLM retry count | Integer (default: 3) |

### Detection Priority

1. **Explicit override**: `RLM_EXECUTION_MODE=native` or `RLM_EXECUTION_MODE=local`
2. **Auto-detect**: Check `DATABRICKS_RUNTIME_VERSION` or `/databricks/spark` existence
3. **Default**: `local` mode

---

## Component Classification

### Components Requiring Dual-Mode Backends (Alternative Implementations)

These components have fundamentally different execution paths for local vs. native:

| Component | Local Mode | Native Mode |
|-----------|-----------|-------------|
| `databricks_repl.py` | REST API to SQL Warehouse | `spark.sql(query)` |
| `unity_catalog.py` | REST API calls | `spark.sql("SHOW ...")` |
| `context_loader.py` | Collects to Python list | Returns Spark DataFrame reference |

### Components Requiring Enhancement (Configuration-Based)

These don't need full alternative implementations—just conditional enhancements:

| Component | Enhancement Needed |
|-----------|-------------------|
| `rlm_repl.py` | Spark injection, DataFrame context support |
| `rlm_tools.py` | Spark-aware strategies, inject `spark` into REPL |
| `llm_bridge.py` | Retry logic, configurable timeout |

### Components Requiring NO Changes (Mode-Agnostic)

These components work identically in both modes:

| Component | Why No Changes |
|-----------|---------------|
| `agent.py` / `root_agent` | Uses tools; tools handle mode internally |
| `agents/rlm_loop.py` | Orchestration logic, no data access |
| `agents/code_executor.py` | Uses `rlm_repl` which handles mode |
| `agents/code_generator.py` | Generates code; no infrastructure calls |
| `agents/completion_checker.py` | Logic only; checks for FINAL patterns |
| `agents/vendor_matcher.py` | Uses tools that handle mode |
| `agents/erp_analyzer.py` | Uses tools that handle mode |
| `agents/view_generator.py` | Uses tools that handle mode |
| `agents/result_formatter.py` | Output formatting only |
| `agents/context_setup.py` | Uses `context_loader` which handles mode |
| `callbacks.py` | Metrics/state management; mode-agnostic |
| `prompts.py` | Static prompt templates |
| `metadata.py` | Data structures only |
| `rlm_state.py` | State management; no I/O |

---

## Change Request: Detailed Implementation Plan

### Phase 1: Foundation (P0)

**Objective**: Create runtime detection and enable dual-mode execution for core data access tools.

#### 1.1 Create `rlm_adk/runtime.py` (New File)

```python
"""Runtime detection and environment configuration for RLM-ADK."""

import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_execution_mode() -> str:
    """Get execution mode: 'native' or 'local'."""
    # Explicit override
    if os.getenv("RLM_EXECUTION_MODE"):
        return os.getenv("RLM_EXECUTION_MODE")
    
    # Auto-detect Databricks runtime
    if is_databricks_runtime():
        return "native"
    
    return "local"

def is_databricks_runtime() -> bool:
    """Detect if code is running on a Databricks cluster."""
    return (
        os.getenv("DATABRICKS_RUNTIME_VERSION") is not None
        or os.path.exists("/databricks/spark")
    )

def get_spark_session():
    """Get SparkSession if in native mode."""
    if get_execution_mode() != "native":
        return None
    from pyspark.sql import SparkSession
    return SparkSession.builder.getOrCreate()

def get_dbutils():
    """Get dbutils if in native mode."""
    if get_execution_mode() != "native":
        return None
    try:
        from pyspark.dbutils import DBUtils
        return DBUtils(get_spark_session())
    except ImportError:
        return None
```

**Acceptance Criteria**:
- [ ] `get_execution_mode()` returns `"native"` when running on a Databricks cluster.
- [ ] `get_execution_mode()` returns `"local"` when running locally.
- [ ] `RLM_EXECUTION_MODE` environment variable can override auto-detection.
- [ ] `get_spark_session()` returns a valid `SparkSession` on cluster, `None` locally.

---

#### 1.2 Update `rlm_adk/tools/databricks_repl.py`

**Change**: Add dual-mode execution—REST API for local, SparkSession for native.

**Dual-Mode Implementation Pattern**:
```python
from rlm_adk.runtime import get_execution_mode, get_spark_session

def execute_sql_query(query: str, tool_context: ToolContext) -> dict:
    if get_execution_mode() == "native":
        return _execute_native_sql(query, get_spark_session())
    return _execute_remote_sql(query, tool_context)

def _execute_native_sql(query: str, spark) -> dict:
    """Execute SQL directly via SparkSession."""
    try:
        df = spark.sql(query)
        columns = df.columns
        data = [row.asDict() for row in df.limit(1000).collect()]
        return {
            "status": "success",
            "columns": columns,
            "data": data,
            "row_count": len(data),
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
        }

def _execute_remote_sql(query: str, tool_context: ToolContext) -> dict:
    """Execute SQL via REST API (existing implementation)."""
    # ... existing REST API code ...
```

| Function | Local Mode | Native Mode |
|----------|-----------|-------------|
| `execute_sql_query()` | REST API to SQL Warehouse | `spark.sql(query)` |
| `execute_python_code()` | Command Execution API | Direct `exec()` with SparkContext |

**Acceptance Criteria**:
- [ ] `execute_sql_query` uses `spark.sql()` when `get_execution_mode() == "native"`.
- [ ] `execute_sql_query` uses REST API when `get_execution_mode() == "local"`.
- [ ] Both paths return the **identical response schema**.

---

#### 1.3 Update `rlm_adk/tools/unity_catalog.py`

**Change**: Add dual-mode execution—REST API for local, Spark SQL for native.

**Dual-Mode Implementation Pattern**:
```python
from rlm_adk.runtime import get_execution_mode, get_spark_session

def list_catalogs(tool_context: ToolContext) -> dict:
    if get_execution_mode() == "native":
        return _native_list_catalogs()
    return _api_list_catalogs(tool_context)

def _native_list_catalogs() -> dict:
    spark = get_spark_session()
    rows = spark.sql("SHOW CATALOGS").collect()
    return {
        "status": "success",
        "catalogs": [{"name": row.catalog} for row in rows],
        "count": len(rows),
    }
```

| Function | Local Mode | Native Mode |
|----------|-----------|-------------|
| `list_catalogs()` | REST `/api/2.1/unity-catalog/catalogs` | `spark.sql("SHOW CATALOGS")` |
| `list_schemas()` | REST API | `spark.sql("SHOW SCHEMAS IN ...")` |
| `list_tables()` | REST API | `spark.sql("SHOW TABLES IN ...")` |
| `list_volumes()` | REST API | REST API or `dbutils.fs.ls()` |
| `get_volume_metadata()` | REST API | `spark.sql("DESCRIBE VOLUME ...")` |
| `read_table_sample()` | Delegates to `execute_sql_query` | `spark.table().limit().collect()` |
| `create_view()` | Delegates to `execute_sql_query` | `spark.sql("CREATE VIEW ...")` |

Note: `list_volumes()` may need to stay REST-based even in native mode (Spark SQL doesn't have `SHOW VOLUMES`), but could use `dbutils`.

**Acceptance Criteria**:
- [ ] All Unity Catalog tools work in both local and native modes.
- [ ] Native mode does NOT require `DATABRICKS_HOST` or `DATABRICKS_TOKEN` environment variables.
- [ ] Both paths return **identical response schemas**.

---

### Phase 2: Context Layer (P1)

**Objective**: Enable Spark DataFrame support in the REPL and context loading.

#### 2.1 Update `rlm_adk/rlm_repl.py`

**Change**: Allow `context` to be either a Python object OR a Spark DataFrame, and inject `spark` into the execution namespace.

**Enhancement**:
```python
from rlm_adk.runtime import get_execution_mode, get_spark_session

class RLMREPLEnvironment:
    def __init__(self, ..., context: Any = None):
        # Context can be:
        # - A Python list/dict (existing behavior)
        # - A Spark DataFrame (new: native mode)
        # - A string table name (new: lazy loading)
        self._context = context
        self._spark = get_spark_session() if get_execution_mode() == "native" else None

    @property
    def context(self):
        # Lazy resolution: if context is a table name string, load it
        if isinstance(self._context, str) and self._spark:
            return self._spark.table(self._context)
        return self._context
    
    def execute_code(self, code: str) -> dict:
        exec_namespace = {
            ...existing...,
            "spark": self._spark,  # Inject when native (None in local mode)
        }
```

**Acceptance Criteria**:
- [ ] `context` can be assigned a Spark DataFrame directly.
- [ ] `context` can be a string like `"catalog.schema.table"` and will be resolved via `spark.table()`.
- [ ] Existing Python list/dict context behavior is unchanged.
- [ ] `spark` is available in code execution namespace when in native mode.

---

#### 2.2 Update `rlm_adk/tools/context_loader.py`

**Change**: In native mode, load context as a Spark DataFrame (not collected to Python list).

**Dual-Mode Implementation**:
```python
from rlm_adk.runtime import get_execution_mode, get_spark_session

def load_vendor_data_to_context(...) -> dict:
    if get_execution_mode() == "native":
        return _load_vendor_data_native(...)
    return _load_vendor_data_local(...)

def _load_vendor_data_native(hospital_chains: list, tool_context: ToolContext) -> dict:
    spark = get_spark_session()
    # Create a temporary view combining all chains
    union_query = " UNION ALL ".join([
        f"SELECT *, '{chain}' as source_chain FROM {chain}.erp_vendors.vendors"
        for chain in hospital_chains
    ])
    df = spark.sql(union_query)
    df.createOrReplaceTempView("rlm_vendor_context")
    
    # Load DataFrame reference (not collected data) into context
    return rlm_load_context(
        context_data=df,  # Spark DataFrame, not list
        context_description=f"Spark DataFrame with {df.count()} vendors",
        tool_context=tool_context,
    )
```

| Function | Local Mode | Native Mode |
|----------|-----------|-------------|
| `load_vendor_data_to_context()` | Collects to Python list | Returns Spark DataFrame reference |
| `_load_chain_data()` | REST SQL → Python list | `spark.table()` → DataFrame |
| `load_query_results_to_context()` | REST SQL → Python list | `spark.sql()` → DataFrame |

**Key Difference:** In native mode, context can remain a Spark DataFrame (lazy evaluation) rather than collected data. This enables processing 10M+ records without OOM.

**Acceptance Criteria**:
- [ ] In native mode, `rlm_vendor_context` temp view is created.
- [ ] Context is a Spark DataFrame reference, not a collected Python list.
- [ ] Works with datasets of 10M+ rows without OOM errors.
- [ ] Local mode behavior is unchanged.

---

#### 2.3 Update `rlm_adk/rlm_tools.py`

**Change**: Add Spark-aware strategies for context operations.

| Function | Enhancement Needed |
|----------|-------------------|
| `rlm_execute_code()` | Inject `spark` into REPL when native |
| `rlm_load_context()` | Accept Spark DataFrame as `context_data`, report schema not just length |
| `rlm_query_context()` | Spark-aware strategies (e.g., use `.rdd.mapPartitions` instead of Python chunking) |

**Acceptance Criteria**:
- [ ] `rlm_execute_code()` provides `spark` in execution namespace when native.
- [ ] `rlm_load_context()` handles Spark DataFrames gracefully.
- [ ] Local mode behavior is unchanged.

---

### Phase 3: Production Hardening (P2)

**Objective**: Add reliability features and deployment infrastructure.

#### 3.1 Verify Network Egress

**Requirement**: The Databricks cluster must have outbound HTTPS access to:
- `generativelanguage.googleapis.com` (Gemini API)
- `oauth2.googleapis.com` (if using OAuth)

**Action Items**:
- [ ] Confirm VPC/Firewall rules allow egress to Google APIs.
- [ ] If using Private Link, configure appropriate endpoints.
- [ ] Document network requirements in `README.md`.

#### 3.2 Update `rlm_adk/llm_bridge.py`

**Change**: Add retry logic and timeout handling suitable for long-running jobs.

**Note**: No alternative implementation needed—Gemini API calls work identically from local or cluster. Only enhancements required.

```python
import os
import time

def _llm_query_fallback(prompt: str, model: str) -> str:
    max_retries = int(os.getenv("RLM_LLM_MAX_RETRIES", "3"))
    timeout = int(os.getenv("RLM_LLM_TIMEOUT_SECONDS", "60"))
    
    for attempt in range(max_retries):
        try:
            # Existing Gemini call with timeout
            ...
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

**Acceptance Criteria**:
- [ ] LLM calls have retry logic with exponential backoff.
- [ ] Timeouts are configurable via environment variable `RLM_LLM_TIMEOUT_SECONDS`.
- [ ] Max retries configurable via `RLM_LLM_MAX_RETRIES`.

---

#### 3.3 Create `rlm_adk/entrypoint.py` (New File)

**Purpose**: Provide a clean entrypoint for Databricks Jobs.

```python
"""Databricks Job entrypoint for RLM-ADK vendor resolution."""

import os
import sys

def main():
    """Run the RLM-ADK root_agent as a Databricks Job."""
    from rlm_adk import root_agent
    # TODO: Integrate with google-adk Runner or custom execution loop
    
    # Example: Run a specific query
    query = os.getenv("RLM_QUERY", "Find duplicate vendors across all hospital chains")
    
    # Execute the agent
    # ... (implementation depends on ADK Runner availability)
    
    print(f"RLM-ADK Job completed. Query: {query}")

if __name__ == "__main__":
    main()
```

#### 3.4 Update `pyproject.toml`

**Add console script entrypoint**:
```toml
[project.scripts]
rlm-adk-job = "rlm_adk.entrypoint:main"
```

**Acceptance Criteria**:
- [ ] `rlm-adk-job` command is available after `pip install`.
- [ ] Job can be invoked via `databricks jobs create` with `python_wheel_task`.

---

#### 3.5 Create `docs/databricks_deployment.md`

**Contents**:
1. Prerequisites (cluster config, network, libraries)
2. Installing `rlm_adk` as a cluster library
3. Configuring environment variables (`GOOGLE_API_KEY`, `RLM_EXECUTION_MODE`)
4. Creating a Databricks Workflow Job
5. Monitoring and logging
6. Troubleshooting common issues (network/auth)
7. Local vs Native mode comparison

**Acceptance Criteria**:
- [ ] Step-by-step deployment guide exists.
- [ ] Example Job JSON configuration is provided.
- [ ] Troubleshooting section covers network/auth issues.
- [ ] Documents `RLM_EXECUTION_MODE` override capability.

---

## Out of Scope

The following items are explicitly **not** part of this change request:
1. Replacing Gemini with Databricks Foundation Model APIs (DBRX, Llama, etc.).
2. Implementing a custom web UI for the agent within Databricks.
3. Multi-tenant isolation (this assumes single-workspace deployment).
4. Real-time streaming ingestion of vendor data.

---

## Testing Strategy

### Unit Tests
- [ ] `test_runtime_detection.py`: Verify `is_databricks_runtime()` with mocked environment.
- [ ] `test_dual_mode_tools.py`: Test each tool in both local and native modes.

### Integration Tests
- [ ] Deploy to a test Databricks workspace.
- [ ] Run `load_vendor_data_to_context` with a real Delta table.
- [ ] Verify `llm_query()` calls reach Gemini from the cluster.
- [ ] Run a full vendor resolution job and verify results.

### Performance Benchmarks
- [ ] Compare execution time: Local (REST API) vs Native (SparkSession).
- [ ] Measure memory usage with 1M+ vendor records.

---

## Implementation Priority

| Component | Changes | Priority | Effort |
|-----------|---------|----------|--------|
| `runtime.py` (new) | Create mode detection module | **P0** | 1 hr |
| `databricks_repl.py` | Add native SQL/Python execution | **P0** | 3 hrs |
| `unity_catalog.py` | Add native catalog operations | **P0** | 3 hrs |
| `context_loader.py` | Add DataFrame context support | **P1** | 2 hrs |
| `rlm_repl.py` | Spark injection, DataFrame context | **P1** | 2 hrs |
| `rlm_tools.py` | Spark-aware strategies | **P1** | 2 hrs |
| `llm_bridge.py` | Retry/timeout config | **P2** | 1 hr |
| `entrypoint.py` (new) | Job entrypoint | **P2** | 2 hrs |
| Tests | Dual-mode test coverage | **P1** | 3 hrs |
| Documentation | Deployment guide | **P2** | 2 hrs |

**Total Estimated Effort**: ~21 hours

### Phase Summary

| Phase | Description | Priority | Total Effort |
|-------|-------------|----------|--------------|
| Phase 1: Foundation | Runtime detection + dual-mode tools | **P0** | 7 hrs |
| Phase 2: Context Layer | DataFrame support + Spark injection | **P1** | 9 hrs |
| Phase 3: Production Hardening | Reliability + deployment | **P2** | 5 hrs |

---

## Key Design Decisions

### What Stays the Same

1. **Tool signatures** — All `FunctionTool` definitions unchanged
2. **Agent orchestration** — `root_agent`, sub-agents, workflows unchanged
3. **Response schemas** — All tools return identical dict structures in both modes
4. **RLM paradigm** — `llm_query()`, `context`, `FINAL` patterns unchanged

### What Changes

1. **Data access layer** — Dual-mode backends in tools (`databricks_repl`, `unity_catalog`, `context_loader`)
2. **Execution environment** — REPL gains `spark` injection capability
3. **Reliability** — LLM bridge gains retry/timeout configuration
4. **Deployment** — New entrypoint for Databricks Jobs

---

## Prompt for Codebase Planning Agent

```
You are a senior software engineer implementing "Dual-Mode Execution" for the rlm_adk library.

Your task is to enable rlm_adk to run in both modes with automatic detection:
1. LOCAL mode (existing) - using REST APIs to communicate with Databricks
2. NATIVE mode (new) - using SparkSession and direct SQL execution on cluster

Key implementation pattern:
- Create rlm_adk/runtime.py with get_execution_mode() and get_spark_session()
- Each dual-mode tool follows: if get_execution_mode() == "native": return _native_impl() else: return _local_impl()
- Both paths MUST return identical response schemas
- RLM_EXECUTION_MODE env var can override auto-detection

Components to modify:
- databricks_repl.py → Add _execute_native_sql(), _execute_native_python()
- unity_catalog.py → Add _native_list_catalogs(), _native_list_schemas(), etc.
- context_loader.py → Return Spark DataFrames instead of collected lists
- rlm_repl.py → Inject spark into execution namespace, support DataFrame context
- rlm_tools.py → Spark-aware strategies for rlm_query_context()
- llm_bridge.py → Add retry/timeout configuration

Components that need NO changes:
- agent.py, all agents/* modules (use tools which handle mode internally)
- callbacks.py, prompts.py, metadata.py, rlm_state.py

Follow AGENTS.md guidelines:
- Fail fast, fail loud - no silent fallbacks
- Both modes return identical schemas
- Small, focused diffs - implement one phase at a time
- Run ruff check --fix after each change

Start with Phase 1 (runtime.py + databricks_repl.py + unity_catalog.py) and proceed sequentially.
```


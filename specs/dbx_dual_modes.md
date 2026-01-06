# RLM-ADK Dual-Mode Architecture: Local vs Native Databricks Execution

## Executive Summary

The `rlm_adk` architecture has a clean separation: **tools/utilities** interact with Databricks infrastructure (need dual-mode), while **agents/orchestration** are infrastructure-agnostic (no changes needed). The environment variable `RLM_EXECUTION_MODE` (or detecting `DATABRICKS_RUNTIME_VERSION`) can drive the switch.

---

## 1. Components Requiring Alternative Implementations (Dual-Mode Backends)

These components have fundamentally different execution paths for local vs. native:

### 1.1 `databricks_repl.py` — HIGH PRIORITY

| Function | Local Mode (Current) | Native Mode (New) |
|----------|---------------------|-------------------|
| `execute_sql_query()` | REST API to SQL Warehouse | `spark.sql(query)` |
| `execute_python_code()` | Command Execution API | Direct `exec()` with SparkContext |
| `_execute_databricks_sql()` | `requests.post()` | `spark.sql().collect()` |
| `_execute_databricks_python()` | `requests.post()` | Local `exec()` in driver |

**Recommended Pattern:**

```python
# databricks_repl.py
def execute_sql_query(query: str, tool_context: ToolContext) -> dict:
    from rlm_adk.runtime import is_databricks_runtime, get_spark_session
    
    if is_databricks_runtime():
        return _execute_native_sql(query, get_spark_session())
    else:
        return _execute_remote_sql(query, tool_context)
```

Both paths return **identical schema** — the tool interface is unchanged.

---

### 1.2 `unity_catalog.py` — HIGH PRIORITY

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

---

### 1.3 `context_loader.py` — MEDIUM PRIORITY

| Function | Local Mode | Native Mode |
|----------|-----------|-------------|
| `load_vendor_data_to_context()` | Collects to Python list | Returns Spark DataFrame reference |
| `_load_chain_data()` | REST SQL → Python list | `spark.table()` → DataFrame |
| `_load_masterdata()` | REST SQL → Python list | `spark.table()` → DataFrame |
| `load_query_results_to_context()` | REST SQL → Python list | `spark.sql()` → DataFrame |

**Key Difference:** In native mode, context can remain a Spark DataFrame (lazy evaluation) rather than collected data. This enables processing 10M+ records without OOM.

---

## 2. Components Requiring Enhancement (Configuration-Based)

These don't need full alternative implementations—just conditional enhancements:

### 2.1 `rlm_repl.py` / `RLMREPLEnvironment` — MEDIUM PRIORITY

**Current state:** Context is always a Python object.

**Enhancement needed:**
1. `context` property should handle Spark DataFrames (lazy resolution)
2. `execute_code()` should inject `spark` into the namespace when native

```python
class RLMREPLEnvironment:
    def __init__(self, ..., context: Any = None):
        self._context = context
        self._spark = None  # Set via runtime detection

    @property
    def context(self):
        # Lazy resolution: if context is a table name string, load it
        if isinstance(self._context, str) and self._spark:
            return self._spark.table(self._context)
        return self._context
    
    def execute_code(self, code: str) -> dict:
        exec_namespace = {
            ...existing...,
            "spark": self._spark,  # Inject when native
        }
```

### 2.2 `rlm_tools.py` — MEDIUM PRIORITY

| Function | Change Needed |
|----------|--------------|
| `rlm_execute_code()` | Inject `spark` into REPL when native |
| `rlm_load_context()` | Accept Spark DataFrame as `context_data`, report schema not just length |
| `rlm_query_context()` | Spark-aware strategies (e.g., use `.rdd.mapPartitions` instead of Python chunking) |

### 2.3 `llm_bridge.py` — LOW PRIORITY

**No alternative implementation needed** — Gemini API calls work identically from local or cluster.

**Enhancements:**
- Add retry with exponential backoff (per spec Phase 5.2)
- Configurable timeout via `RLM_LLM_TIMEOUT_SECONDS`

```python
def _llm_query_fallback(prompt: str, model: str) -> str:
    max_retries = int(os.getenv("RLM_LLM_MAX_RETRIES", "3"))
    timeout = int(os.getenv("RLM_LLM_TIMEOUT_SECONDS", "60"))
    
    for attempt in range(max_retries):
        try:
            # existing Gemini call with timeout
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
```

---

## 3. Components Requiring NO Changes (Mode-Agnostic)

### 3.1 All Agent Definitions

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

### 3.2 Supporting Modules

| Component | Why No Changes |
|-----------|---------------|
| `callbacks.py` | Metrics/state management; mode-agnostic |
| `prompts.py` | Static prompt templates |
| `metadata.py` | Data structures only |
| `rlm_state.py` | State management; no I/O |
| `testing.py` | Test utilities |

---

## 4. Proposed Implementation Architecture

### 4.1 Create `rlm_adk/runtime.py` (New File)

This is the **single source of truth** for mode detection:

```python
"""Runtime detection and environment configuration."""

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
    """Detect if running on Databricks cluster."""
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

### 4.2 Tool Implementation Pattern

Each dual-mode tool follows this pattern:

```python
# unity_catalog.py
from rlm_adk.runtime import get_execution_mode, get_spark_session

def list_catalogs(tool_context: ToolContext) -> dict:
    if get_execution_mode() == "native":
        return _native_list_catalogs()
    return _api_list_catalogs(...)

def _native_list_catalogs() -> dict:
    spark = get_spark_session()
    rows = spark.sql("SHOW CATALOGS").collect()
    return {
        "status": "success",
        "catalogs": [{"name": row.catalog} for row in rows],
        "count": len(rows),
    }
```

---

## 5. Updated Effort Estimates

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

**Total:** ~21 hours

---

## 6. Key Design Decisions

### 6.1 Environment Variable Schema

| Variable | Purpose | Values |
|----------|---------|--------|
| `RLM_EXECUTION_MODE` | Force mode override | `local`, `native`, or unset (auto-detect) |
| `DATABRICKS_RUNTIME_VERSION` | Auto-detection marker | Set by Databricks runtime |
| `RLM_LLM_TIMEOUT_SECONDS` | LLM call timeout | Integer (default: 60) |
| `RLM_LLM_MAX_RETRIES` | LLM retry count | Integer (default: 3) |

### 6.2 What Stays the Same

1. **Tool signatures** — All `FunctionTool` definitions unchanged
2. **Agent orchestration** — `root_agent`, sub-agents, workflows unchanged
3. **Response schemas** — All tools return identical dict structures
4. **RLM paradigm** — `llm_query()`, `context`, `FINAL` patterns unchanged

---

## 7. Visual Architecture Comparison

### Local Mode (Current)

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

### Native Mode (Target)

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

---

## 8. Implementation Phases

### Phase 1: Foundation (P0)
1. Create `rlm_adk/runtime.py` with mode detection
2. Update `databricks_repl.py` with dual-mode SQL/Python execution
3. Update `unity_catalog.py` with dual-mode catalog operations

### Phase 2: Context Layer (P1)
4. Update `context_loader.py` for DataFrame support
5. Update `rlm_repl.py` with Spark injection
6. Update `rlm_tools.py` with Spark-aware strategies
7. Add dual-mode test coverage

### Phase 3: Production Hardening (P2)
8. Add retry/timeout to `llm_bridge.py`
9. Create `entrypoint.py` for Databricks Jobs
10. Write deployment documentation

---

## 9. Testing Strategy

### Unit Tests
- `test_runtime_detection.py`: Mock `DATABRICKS_RUNTIME_VERSION` and verify detection
- `test_dual_mode_tools.py`: Test each tool in both modes with mocked backends

### Integration Tests (Native Mode)
- Deploy to test Databricks workspace
- Run `load_vendor_data_to_context` with real Delta table
- Verify `llm_query()` calls reach Gemini from cluster
- Run full vendor resolution job

### Performance Benchmarks
- Compare execution time: Local (REST API) vs Native (SparkSession)
- Measure memory usage with 1M+ vendor records
- Verify no OOM with DataFrame context


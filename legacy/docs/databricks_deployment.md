# RLM-ADK Databricks Deployment Guide

This guide covers deploying RLM-ADK for native execution on Databricks clusters.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Environment Configuration](#environment-configuration)
4. [Creating a Databricks Workflow Job](#creating-a-databricks-workflow-job)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Troubleshooting](#troubleshooting)
7. [Local vs Native Mode Comparison](#local-vs-native-mode-comparison)

---

## Prerequisites

### Cluster Requirements

- **Databricks Runtime**: 14.0+ (with Apache Spark 3.4+)
- **Node Type**: Recommended `Standard_DS3_v2` or equivalent (4 cores, 14GB RAM minimum)
- **Python Version**: 3.11+

### Network Configuration

The cluster must have outbound HTTPS access to:

| Endpoint | Purpose |
|----------|---------|
| `generativelanguage.googleapis.com` | Gemini API for LLM calls |
| `oauth2.googleapis.com` | Google OAuth (if using service account) |
| `pypi.org` | Package installation |

If using Private Link or VPC, ensure these endpoints are allowlisted.

### Required Secrets

Store these in Databricks Secrets or as cluster environment variables:

| Secret | Description |
|--------|-------------|
| `GOOGLE_API_KEY` | Gemini API key for LLM inference |

---

## Installation

### Option 1: Install as Cluster Library

1. Navigate to **Compute** → Select your cluster → **Libraries**
2. Click **Install New** → **PyPI**
3. Enter: `rlm[rlm-adk,native]`
4. Click **Install**

### Option 2: Install in Notebook

```python
%pip install rlm[rlm-adk,native]
dbutils.library.restartPython()
```

### Option 3: Init Script

Create an init script at `/dbfs/init-scripts/install-rlm.sh`:

```bash
#!/bin/bash
pip install rlm[rlm-adk,native]
```

---

## Environment Configuration

### Required Environment Variables

Set these in your cluster configuration or job parameters:

```bash
# Required
GOOGLE_API_KEY=your-gemini-api-key

# Optional (with defaults)
RLM_EXECUTION_MODE=native          # Force native mode (auto-detected on Databricks)
RLM_LLM_TIMEOUT_SECONDS=60         # LLM call timeout
RLM_LLM_MAX_RETRIES=3              # LLM retry attempts

# Job-specific
RLM_QUERY="Find duplicate vendors across hospital chains"
RLM_HOSPITAL_CHAINS="hospital_chain_alpha,hospital_chain_beta"
RLM_INCLUDE_MASTERDATA=true
```

### Setting Environment Variables in Cluster

1. Go to **Compute** → Select cluster → **Configuration**
2. Expand **Advanced Options** → **Spark** → **Environment Variables**
3. Add your variables

### Using Databricks Secrets

```python
# In your notebook or job
import os
os.environ["GOOGLE_API_KEY"] = dbutils.secrets.get(scope="rlm-adk", key="google-api-key")
```

---

## Creating a Databricks Workflow Job

### Using the UI

1. Navigate to **Workflows** → **Create Job**
2. Configure the task:

| Setting | Value |
|---------|-------|
| Task name | `vendor-resolution` |
| Type | Python wheel task |
| Package name | `rlm` |
| Entry point | `rlm-adk-job` |
| Cluster | Select your configured cluster |

### Using the Jobs API

```json
{
  "name": "RLM-ADK Vendor Resolution",
  "tasks": [
    {
      "task_key": "vendor_resolution",
      "python_wheel_task": {
        "package_name": "rlm",
        "entry_point": "rlm-adk-job",
        "parameters": []
      },
      "libraries": [
        {"pypi": {"package": "rlm[rlm-adk,native]"}}
      ],
      "new_cluster": {
        "spark_version": "14.3.x-scala2.12",
        "node_type_id": "Standard_DS3_v2",
        "num_workers": 0,
        "spark_conf": {
          "spark.databricks.cluster.profile": "singleNode"
        },
        "spark_env_vars": {
          "GOOGLE_API_KEY": "{{secrets/rlm-adk/google-api-key}}",
          "RLM_HOSPITAL_CHAINS": "hospital_chain_alpha,hospital_chain_beta,hospital_chain_gamma",
          "RLM_INCLUDE_MASTERDATA": "true"
        }
      }
    }
  ],
  "schedule": {
    "quartz_cron_expression": "0 0 2 ? * MON",
    "timezone_id": "America/New_York"
  }
}
```

### Using Databricks CLI

```bash
databricks jobs create --json-file job-config.json
```

---

## Monitoring and Logging

### Job Output

The RLM-ADK job outputs structured logs:

```
============================================================
RLM-ADK Job Starting
Timestamp: 2024-01-15T02:00:00
============================================================

Runtime Info:
  execution_mode: native
  spark_available: True
  ...

------------------------------------------------------------
Phase 1: Loading Context
------------------------------------------------------------
  Status: success
  Total Vendors: 15000
  ...

------------------------------------------------------------
Phase 2: Executing Query
------------------------------------------------------------
  Strategy: spark_sql
  ...

============================================================
RLM-ADK Job Completed Successfully
============================================================

__RESULT_JSON__: {"status": "success", ...}
```

### Accessing Results

The final JSON result is printed with `__RESULT_JSON__:` prefix for easy parsing:

```python
# Parse job output
import json
for line in job_output.split("\n"):
    if line.startswith("__RESULT_JSON__:"):
        result = json.loads(line.replace("__RESULT_JSON__:", "").strip())
        print(result)
```

---

## Troubleshooting

### Common Issues

#### 1. "PySpark is required for native mode execution"

**Cause**: PySpark not installed on the cluster.

**Solution**: Install with `rlm[native]` or ensure cluster has Spark runtime.

#### 2. "Failed to get SparkSession"

**Cause**: Code running outside Spark context.

**Solution**: Ensure code runs on a cluster, not in a notebook with detached compute.

#### 3. LLM API Timeout

**Cause**: Network issues or API overload.

**Solution**:
- Check network egress to `generativelanguage.googleapis.com`
- Increase `RLM_LLM_TIMEOUT_SECONDS`
- Increase `RLM_LLM_MAX_RETRIES`

#### 4. "Invalid access token"

**Cause**: Expired or invalid `GOOGLE_API_KEY`.

**Solution**: Regenerate API key and update Databricks secret.

#### 5. Unity Catalog Access Denied

**Cause**: Cluster/user lacks permissions to read tables.

**Solution**: Grant SELECT permission on required catalogs/schemas.

---

## Local vs Native Mode Comparison

| Aspect | Local Mode | Native Mode |
|--------|------------|-------------|
| **Execution Location** | Developer workstation | Databricks cluster |
| **Data Access** | REST API to SQL Warehouse | Direct SparkSession |
| **Data Transfer** | Downloads to local memory | Data stays in cluster |
| **Scale** | Limited by local RAM | Distributed Spark processing |
| **Authentication** | DATABRICKS_TOKEN required | Ambient credentials |
| **Scheduling** | Manual execution | Databricks Workflow Jobs |
| **Security** | Data leaves Databricks | Data stays in perimeter |
| **Use Case** | Development, testing | Production, large-scale |

### When to Use Each Mode

**Use Local Mode when:**
- Developing and testing agent behavior
- Working with small datasets (< 100K records)
- Debugging issues interactively
- Running one-off analyses

**Use Native Mode when:**
- Processing production datasets (millions of records)
- Running scheduled automated pipelines
- Data must stay within security perimeter
- Leveraging Spark for distributed joins/aggregations

---

## Example: Complete Workflow

```python
# 1. Verify runtime
from rlm_adk.runtime import get_runtime_info
print(get_runtime_info())

# 2. Load context
from rlm_adk.tools.context_loader import load_vendor_data_to_context
from unittest.mock import MagicMock

ctx = MagicMock()
ctx.state = {}

result = load_vendor_data_to_context(
    hospital_chains=["hospital_chain_alpha", "hospital_chain_beta"],
    include_masterdata=True,
    tool_context=ctx,
)
print(f"Loaded {result['total_vendors']} vendors")

# 3. Query with RLM decomposition
from rlm_adk.tools.rlm_tools import rlm_query_context

answer = rlm_query_context(
    query="Find all duplicate vendors based on tax ID",
    strategy="spark_sql",  # Uses Spark in native mode
    tool_context=ctx,
)
print(answer["answer"])
```

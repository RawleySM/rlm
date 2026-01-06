"""Runtime detection and environment configuration for RLM-ADK.

This module provides the single source of truth for execution mode detection,
enabling dual-mode execution: Local Orchestration (REST APIs) and Native
Execution (direct SparkSession on Databricks clusters).

Environment Variables:
    RLM_EXECUTION_MODE: Force mode override ('local', 'native', or unset for auto-detect)
    DATABRICKS_RUNTIME_VERSION: Auto-detection marker (set by Databricks runtime)
    RLM_LLM_TIMEOUT_SECONDS: LLM call timeout (default: 60)
    RLM_LLM_MAX_RETRIES: LLM retry count (default: 3)

Detection Priority:
    1. Explicit override via RLM_EXECUTION_MODE
    2. Auto-detect via DATABRICKS_RUNTIME_VERSION or /databricks/spark existence
    3. Default to 'local' mode
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


@lru_cache(maxsize=1)
def get_execution_mode() -> str:
    """Get the current execution mode: 'native' or 'local'.

    Uses cached result for performance. The mode is determined once
    and reused for the lifetime of the process.

    Returns:
        'native' if running on Databricks cluster, 'local' otherwise.

    Examples:
        >>> mode = get_execution_mode()
        >>> if mode == 'native':
        ...     spark = get_spark_session()
        ...     df = spark.sql("SELECT * FROM table")
    """
    # Priority 1: Explicit override via environment variable
    explicit_mode = os.getenv("RLM_EXECUTION_MODE")
    if explicit_mode:
        if explicit_mode.lower() in ("native", "local"):
            return explicit_mode.lower()
        raise ValueError(
            f"Invalid RLM_EXECUTION_MODE '{explicit_mode}'. Must be 'native' or 'local'."
        )

    # Priority 2: Auto-detect Databricks runtime
    if is_databricks_runtime():
        return "native"

    # Priority 3: Default to local mode
    return "local"


def is_databricks_runtime() -> bool:
    """Detect if code is running on a Databricks cluster.

    Checks for Databricks-specific environment markers:
    1. DATABRICKS_RUNTIME_VERSION environment variable
    2. /databricks/spark directory existence

    Returns:
        True if running on Databricks cluster, False otherwise.
    """
    # Check environment variable (most reliable)
    if os.getenv("DATABRICKS_RUNTIME_VERSION") is not None:
        return True

    # Check for Databricks filesystem marker
    if os.path.exists("/databricks/spark"):
        return True

    return False


def get_spark_session() -> SparkSession | None:
    """Get SparkSession if in native mode.

    In native mode, returns the active SparkSession from the Databricks
    cluster. In local mode, returns None.

    Returns:
        SparkSession instance in native mode, None in local mode.

    Raises:
        RuntimeError: If in native mode but SparkSession cannot be obtained.

    Examples:
        >>> spark = get_spark_session()
        >>> if spark:
        ...     df = spark.sql("SELECT * FROM catalog.schema.table")
    """
    if get_execution_mode() != "native":
        return None

    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        return spark
    except ImportError as e:
        raise RuntimeError(
            "PySpark is required for native mode execution. "
            "Install with: pip install pyspark"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to get SparkSession: {e}") from e


def get_dbutils():
    """Get dbutils if in native mode.

    dbutils provides utilities for working with Databricks features like
    secrets, widgets, and filesystem operations.

    Returns:
        dbutils instance in native mode, None in local mode.

    Examples:
        >>> dbutils = get_dbutils()
        >>> if dbutils:
        ...     secret = dbutils.secrets.get(scope="my-scope", key="my-key")
    """
    if get_execution_mode() != "native":
        return None

    try:
        # Method 1: Try getting dbutils from IPython (Databricks notebooks)
        try:
            from IPython import get_ipython

            ipython = get_ipython()
            if ipython is not None:
                dbutils = ipython.user_ns.get("dbutils")
                if dbutils is not None:
                    return dbutils
        except (ImportError, AttributeError):
            pass

        # Method 2: Try importing DBUtils directly (Databricks runtime)
        try:
            from pyspark.dbutils import DBUtils

            spark = get_spark_session()
            if spark:
                return DBUtils(spark)
        except ImportError:
            pass

        # Method 3: Try getting from SparkContext._jvm
        try:
            spark = get_spark_session()
            if spark and hasattr(spark, "_jvm"):
                return spark._jvm.com.databricks.dbutils_v1.DBUtilsHolder.dbutils()
        except Exception:
            pass

        return None

    except Exception:
        return None


def get_llm_timeout_seconds() -> int:
    """Get the configured LLM call timeout in seconds.

    Returns:
        Timeout value from RLM_LLM_TIMEOUT_SECONDS env var, or 60 (default).
    """
    try:
        return int(os.getenv("RLM_LLM_TIMEOUT_SECONDS", "60"))
    except ValueError:
        return 60


def get_llm_max_retries() -> int:
    """Get the configured maximum LLM retry count.

    Returns:
        Max retries from RLM_LLM_MAX_RETRIES env var, or 3 (default).
    """
    try:
        return int(os.getenv("RLM_LLM_MAX_RETRIES", "3"))
    except ValueError:
        return 3


def clear_execution_mode_cache() -> None:
    """Clear the cached execution mode.

    Useful for testing or when environment changes mid-process.
    After clearing, the next call to get_execution_mode() will
    re-detect the mode.
    """
    get_execution_mode.cache_clear()


def get_runtime_info() -> dict:
    """Get comprehensive runtime information for debugging.

    Returns:
        Dict with execution mode, detection details, and configuration.
    """
    return {
        "execution_mode": get_execution_mode(),
        "is_databricks_runtime": is_databricks_runtime(),
        "databricks_runtime_version": os.getenv("DATABRICKS_RUNTIME_VERSION"),
        "explicit_mode_override": os.getenv("RLM_EXECUTION_MODE"),
        "spark_available": get_spark_session() is not None,
        "dbutils_available": get_dbutils() is not None,
        "llm_timeout_seconds": get_llm_timeout_seconds(),
        "llm_max_retries": get_llm_max_retries(),
    }

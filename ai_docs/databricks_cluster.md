# Deploying RLM-ADK on Databricks Clusters

This guide details how to package and deploy the `rlm_adk` agent to run natively on Databricks clusters. This deployment strategy leverages the "Native Mode" architecture described in `@specs/dbx_dual_modes.md`, enabling direct access to `SparkSession` for high-performance context loading and SQL execution.

## Prerequisites

*   **Databricks CLI**: Installed and authenticated (`databricks auth login`).
*   **Build Tools**: `uv` or standard Python build tools (`build`, `wheel`).
*   **Unity Catalog**: (Recommended) A volume for storing the agent wheel, e.g., `/Volumes/dev/rlm_agents/wheels`.

## 1. Packaging the Agent

The agent must be packaged as a Python Wheel (`.whl`) to be installed on the Databricks cluster.

### Using `uv` (Recommended)

```bash
# From the project root (where pyproject.toml is located)
uv build
```

This will generate a wheel file in the `dist/` directory, e.g., `dist/rlm_adk-0.1.0-py3-none-any.whl`.

### Using standard tools

```bash
pip install build
python -m build
```

## 2. Uploading the Wheel

Upload the generated wheel to a location accessible by your Databricks cluster. We recommend using a Unity Catalog Volume for better governance, but DBFS works as well.

### Option A: Unity Catalog Volume (Recommended)

```bash
# Create volume if it doesn't exist (one-time setup)
# databricks unity-catalog volumes create --catalog dev --schema rlm_agents --name wheels

# Upload the wheel
databricks fs cp dist/rlm_adk-0.1.0-py3-none-any.whl dbfs:/Volumes/dev/rlm_agents/wheels/rlm_adk-0.1.0-py3-none-any.whl
```

### Option B: DBFS

```bash
databricks fs cp dist/rlm_adk-0.1.0-py3-none-any.whl dbfs:/FileStore/jars/rlm_adk-0.1.0-py3-none-any.whl
```

## 3. Creating the Databricks Job

Configure a Databricks Job to execute the agent. We use the **Python Wheel** task type.

### Job Configuration

1.  **Task Name**: `Run_RLM_Agent`
2.  **Type**: `Python Wheel`
3.  **Package Name**: `rlm_adk` (Match the name in `pyproject.toml`)
4.  **Entry Point**: `rlm-run` (or your defined entry point console script)
    *   *Note: Ensure your `pyproject.toml` defines this script. See [Entry Point Setup](#entry-point-setup) below.*
5.  **Compute**:
    *   **Runtime Version**: Recommended **Databricks Runtime 13.3 LTS ML** or higher. The ML runtime includes many common dependencies.
    *   **Node Type**: Choose based on workload (e.g., `Standard_DS3_v2`).
6.  **Dependent Libraries**:
    *   Click **+ Add** -> **Python Wheel**
    *   Path: `dbfs:/Volumes/dev/rlm_agents/wheels/rlm_adk-0.1.0-py3-none-any.whl` (or your DBFS path).
7.  **Environment Variables**:
    *   `RLM_EXECUTION_MODE`: `native` (Optional, auto-detected, but good for clarity).
    *   `GOOGLE_API_KEY`: `<your-gemini-api-key>` (Required for LLM access).
    *   *Note: For production, use Databricks Secrets reference syntax: `{{secrets/scope/key}}`*.

### Entry Point Setup

Ensure your `pyproject.toml` has a script entry point defined:

```toml
[project.scripts]
rlm-run = "rlm_adk.entrypoint:main"
```

And create the `rlm_adk/entrypoint.py` file if it doesn't exist:

```python
# rlm_adk/entrypoint.py
import os
import sys
from rlm_adk.agent import root_agent
from rlm_adk.runtime import get_execution_mode

def main():
    print(f"Starting RLM Agent in {get_execution_mode()} mode...")
    
    # Parse arguments if needed
    user_query = sys.argv[1] if len(sys.argv) > 1 else "Run default analysis"
    
    # Initialize and run agent
    # ... (Add your runner logic here, e.g. Runner(agent=root_agent).run_sync(...))
    
    print("Agent execution complete.")

if __name__ == "__main__":
    main()
```

## 4. Execution & Monitoring

1.  **Run the Job**: Click "Run Now" in the Databricks UI or use the CLI:
    ```bash
    databricks jobs run-now --job-id <JOB_ID> --python-params '["Analyze vendor spend for Q3"]'
    ```
2.  **Monitor Output**:
    *   Go to the **Job Run** details.
    *   Click on the **Task Run**.
    *   View the **Driver Logs** (Standard Output) to see the agent's progress and responses.

## Troubleshooting

*   **Module Not Found**: Ensure the wheel path in "Dependent Libraries" is correct and the cluster has permissions to read it.
*   **Spark Context Error**: If `get_spark_session()` fails, ensure you are running in a Databricks Job context, not just a local Python script on the driver node without Spark connected.
*   **Authentication Errors**: Verify `GOOGLE_API_KEY` is set correctly in the job environment variables.

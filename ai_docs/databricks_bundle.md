# Deploying `rlm_adk` to Databricks with Asset Bundles and Databricks Connect

This guide details how to deploy the `rlm_adk` Python project to a Databricks workspace using Databricks Asset Bundles (DABs) and how to leverage the Databricks Connect IDE extension for development.

## Prerequisites

1.  **Databricks CLI**: Ensure you have the latest version of the Databricks CLI installed (version 0.200.0 or higher).
2.  **Databricks Workspace**: Access to a Databricks workspace with permissions to create jobs and deploy artifacts.
3.  **VS Code**: Visual Studio Code installed.
4.  **Databricks Extension for VS Code**: Install the official Databricks extension from the VS Code Marketplace.
5.  **Python Environment**: A local Python environment (e.g., venv, conda) with `rlm_adk` dependencies installed.

## 1. Setting up Databricks Asset Bundles (DABs)

DABs allow you to define your infrastructure and code as a bundle.

### Initialize the Bundle

Run the following command in the root of your `rlm_adk` project to initialize a new bundle. You can also manually create the `databricks.yml` file.

```bash
databricks bundle init
```

Follow the prompts to select the "Python" template if available, or start from scratch.

### Configure `databricks.yml`

Create or update the `databricks.yml` file in your project root. This file defines the bundle configuration, resources, and targets.

```yaml
bundle:
  name: rlm_adk_bundle

include:
  - resources/*.yml

targets:
  dev:
    mode: development
    default: true
    workspace:
      host: <your-databricks-workspace-url>
      root_path: /Users/${workspace.current_user.userName}/.bundle/${bundle.name}/${bundle.target}
    run_as:
      user_name: ${workspace.current_user.userName}

  prod:
    mode: production
    workspace:
      host: <your-databricks-workspace-url>
      root_path: /Shared/bundles/${bundle.name}/${bundle.target}
    run_as:
      service_principal_name: <service-principal-app-id>
```

### Define Resources

Create a `resources/rlm_adk_job.yml` file to define the job that will run your code.

```yaml
resources:
  jobs:
    rlm_adk_job:
      name: rlm_adk_job
      tasks:
        - task_key: main_task
          job_cluster_key: job_cluster
          python_wheel_task:
            package_name: rlm_adk
            entry_point: run_agent  # Ensure this entry point is defined in setup.py/pyproject.toml
          libraries:
            - whl: ../dist/*.whl

      job_clusters:
        - job_cluster_key: job_cluster
          new_cluster:
            spark_version: 13.3.x-scala2.12
            node_type_id: Standard_DS3_v2
            num_workers: 1
```

## 2. Python Packaging

Ensure `rlm_adk` is set up as a proper Python package.

1.  **`pyproject.toml` or `setup.py`**: Verify you have the package configuration.
2.  **Entry Point**: Define a console script entry point if you want to run it directly as a task.

Example `pyproject.toml` snippet:

```toml
[project.scripts]
run_agent = "rlm_adk.main:main"
```

## 3. Deploying the Bundle

### Validate the Bundle

Check your configuration for errors:

```bash
databricks bundle validate
```

### Deploy to Development Target

Deploy the code and configuration to your `dev` target:

```bash
databricks bundle deploy -t dev
```

This command builds your Python wheel, uploads it to the workspace, and updates the defined job.

### Run the Job

Execute the deployed job:

```bash
databricks bundle run rlm_adk_job -t dev
```

## 4. Using Databricks Connect in VS Code

Databricks Connect allows you to debug and iterate on code locally while executing on a remote Databricks cluster.

### Configure the Extension

1.  Open the **Databricks** extension in VS Code.
2.  **Configure Sync**: Set up the extension to sync your local `rlm_adk` directory to a workspace directory.
3.  **Select Cluster**: Attach to a running Databricks cluster (ensure it meets the Databricks Connect requirements, e.g., DBR 13.3+).

### Debugging

1.  Open a Python file in `rlm_adk`.
2.  Use the "Run on Databricks" or "Debug on Databricks" options provided by the extension.
3.  The code will run locally but interact with the Spark session on the remote cluster.

### Development Workflow

1.  **Code**: Make changes locally in VS Code.
2.  **Sync**: The extension automatically syncs changes to the workspace.
3.  **Run/Debug**: Execute the code using Databricks Connect to verify behavior on the cluster.
4.  **Deploy**: Once satisfied, use `databricks bundle deploy` to push the stable version as a job.

## Summary

*   **`databricks.yml`**: The source of truth for your deployment configuration.
*   **`databricks bundle deploy`**: Pushes your code and infrastructure changes.
*   **VS Code Extension**: Enables rapid iteration and debugging via Databricks Connect.

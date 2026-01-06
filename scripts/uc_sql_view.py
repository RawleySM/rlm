#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "databricks-sdk",
#   "click",
#   "ulid-py",
#   "typing_extensions",
# ]
# ///
import os
import ulid
import logging
import time
import base64
import sys
import io
import re
import click
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
try:
    from uc_sql_profile import profile_table
except ImportError:
    # If running from root, try relative import or just assume it's available in path
    try:
        from src.vendor_er.optimization.tools.uc_sql_profile import profile_table
    except ImportError:
         print("Warning: Could not import profile_table. Profiling will be skipped.")
         profile_table = None

try:
    from uc_sql_fill import calculate_fill_rate
except ImportError:
    try:
        from src.vendor_er.optimization.tools.uc_sql_fill import calculate_fill_rate
    except ImportError:
        print("Warning: Could not import calculate_fill_rate. Fill rate calculation will be skipped.")
        calculate_fill_rate = None

# Configuration
JOB_ID = 90279521676136
WORKSPACE_BASE_PATH = "dbfs:/FileStore/task_sql"
PROFILE = "rstanhope"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("create_view_from_local")

def get_db_client():
    """Get Databricks Workspace Client."""
    try:
        logger.info(f"Authenticating with profile: {PROFILE}")
        return WorkspaceClient(profile=PROFILE)
    except Exception as e:
        logger.error(f"Failed to create Databricks client: {e}")
        logger.info("Ensure you have run 'databricks auth login --profile rstanhope'")
        raise

def generate_artifact_content(sql_content: str) -> str:
    """Generate the Python artifact content wrapping the SQL."""
    # Escape triple quotes in SQL to avoid syntax errors in the generated Python string
    safe_sql = sql_content.replace('"""', '\"\"\"')
    
    return f"""
from pyspark.sql import SparkSession
import logging
import sys

# Configure remote logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("task_sql_runner")

def execute_view_logic(spark):
    logger.info("Starting execution of dynamic SQL view logic...")
    
    sql_query = \"\"\"{safe_sql}\"\"\" 
    
    try:
        logger.info("Executing SQL query...")
        # Execute the query
        df = spark.sql(sql_query)
        
        # Action to force execution and get some stats
        # We assume the SQL is DDL (CREATE VIEW) or DML that returns something.
        # If it's just CREATE VIEW, count() might not work or be empty, 
        # but let's try to see if it returns results (like 'OK') or if we need to just run it.
        # For CREATE VIEW, it usually returns an empty DataFrame or similar.
        # Let's check if it is a SELECT or DDL.
        
        if sql_query.strip().upper().startswith("SELECT"):
            count = df.count()
            logger.info(f"Query executed. Row count: {{count}}")
            df.show(n=20, truncate=False)
        else:
            # For DDL/DML, we might need to iterate to force execution if it's lazy, 
            # though spark.sql() for DDL is usually eager.
            # However, spark.sql() returns a DataFrame.
            # Calling collect() ensures execution.
            logger.info("Executing non-SELECT statement...")
            df.show(n=20, truncate=False)
            results = df.collect()
            logger.info("SQL executed successfully.")
            
    except Exception as e:
        logger.error(f"Error executing SQL: {{e}}")
        raise e

if __name__ == "__main__":
    spark = SparkSession.builder.appName("DynamicSQLRunner").getOrCreate()
    execute_view_logic(spark)
"""

def extract_table_name(sql_content):
    """Extracts the table/view name from a CREATE VIEW/TABLE statement."""
    # Look for CREATE [OR REPLACE] [VIEW|TABLE] [IF NOT EXISTS] catalog.schema.table
    pattern = r"(?i)CREATE\s+(?:OR\s+REPLACE\s+)?(?:VIEW|TABLE)\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)"
    match = re.search(pattern, sql_content)
    if match:
        return match.group(1)
    return None

def monitor_run(w: WorkspaceClient, run_id: int):
    """Monitor the job run and stream logs."""
    logger.info(f"Monitoring run ID: {run_id}")
    
    printed_length = 0
    
    while True:
        run = w.jobs.get_run(run_id)
        state = run.state.life_cycle_state
        
        # Determine the correct run ID to query for logs (Task Run ID vs Job Run ID)
        log_run_id = run_id
        if run.tasks:
            for task in run.tasks:
                if task.task_key == "create_view": # Target our specific task if possible
                    log_run_id = task.run_id
                    break
            else:
                # Fallback to first task if 'create_view' not found
                if len(run.tasks) > 0:
                    log_run_id = run.tasks[0].run_id

        # Attempt to get logs
        try:
            output = w.jobs.get_run_output(log_run_id)
            
            if output.logs:
                # We prioritize 'logs' field if available, or 'logs_overload' if that's where it is.
                # The SDK usually puts stdout/stderr in logs.
                # Assuming simple string concatenation for now.
                current_logs = output.logs
                if len(current_logs) > printed_length:
                    new_logs = current_logs[printed_length:]
                    print(new_logs, end='', flush=True)
                    printed_length = len(current_logs)
            
            if output.error:
                 # If there is an error field, log it, but don't stop unless the job state is terminal
                 pass

        except Exception as e:
            # Logs might not be available immediately or API might flake
            pass

        if state in [jobs.RunLifeCycleState.TERMINATED, jobs.RunLifeCycleState.SKIPPED]:
            if run.state.result_state == jobs.RunResultState.SUCCESS:
                logger.info(f"Run {run_id} succeeded.")
                return True
            else:
                logger.error(f"Run {run_id} failed with state: {run.state.result_state}")
                logger.error(f"State message: {run.state.state_message}")
                raise Exception(f"Job run failed: {run.state.result_state}")
        
        if state in [jobs.RunLifeCycleState.INTERNAL_ERROR]:
             logger.error(f"Run {run_id} failed with internal error.")
             raise Exception(f"Job run internal error: {run.state.state_message}")

        time.sleep(5)

@click.command()
@click.argument('sql_input')
def main(sql_input):
    """
    Automate Databricks Job Execution for SQL Views. 
    
    SQL_INPUT can be a file path or a raw SQL string.
    """
    
    # 1. Generate ULID and Module Name
    current_ulid = ulid.new().str
    task_module_name = f"view_{current_ulid}"
    logger.info(f"Generated task module name: {task_module_name}")
    
    # 2. Read SQL
    is_input_file = False
    if os.path.exists(sql_input):
        logger.info(f"Reading SQL from file: {sql_input}")
        is_input_file = True
        with open(sql_input, 'r') as f:
            sql_content = f.read()
    else:
        logger.info("Treating input as raw SQL string")
        sql_content = sql_input
        
    if not sql_content.strip():
        logger.error("Empty SQL content.")
        return

    # 3. Generate Artifact
    artifact_content = generate_artifact_content(sql_content)
    artifact_path = f"{WORKSPACE_BASE_PATH}/{task_module_name}.py"
    
    w = get_db_client()
    
    # 4. Upload Artifact
    logger.info(f"Uploading artifact to {artifact_path}...")
    try:
        w.dbfs.upload(
            path=artifact_path,
            src=io.BytesIO(artifact_content.encode('utf-8')),
            overwrite=True
        )
        logger.info("Upload successful.")
    except Exception as e:
        logger.error(f"Failed to upload artifact: {e}")
        return

    # 5. Trigger Job
    logger.info(f"Triggering Job ID {JOB_ID} with task_module_uuid={task_module_name}...")
    try:
        run = w.jobs.run_now(
            job_id=JOB_ID,
            job_parameters={"task_module_uuid": task_module_name}
        )
        run_id = run.run_id
        logger.info(f"Job started. Run ID: {run_id}")
    except Exception as e:
        logger.error(f"Failed to trigger job: {e}")
        return

    # 6. Monitor & Post-Process
    try:
        success = monitor_run(w, run_id)
        if success:
            logger.info("Job execution successful. processing artifacts...")
            
            # Retrieve and save the SQL that was actually executed
            try:
                logger.info(f"Retrieving execution artifact from {artifact_path}...")
                with w.dbfs.download(artifact_path) as response:
                    downloaded_content = response.read().decode('utf-8')
                
                # Extract SQL from the wrapper
                # Pattern matches: sql_query = """<content>"""
                match = re.search(r'sql_query\s*=\s*"""(.*?)"""', downloaded_content, re.DOTALL)
                if match:
                    retrieved_sql = match.group(1)
                    # Unescape triple quotes
                    retrieved_sql = retrieved_sql.replace('\"\"\"', '"""')
                    
                    # Write to local queries directory
                    local_query_dir = "src/vendor_er/optimization/queries"
                    os.makedirs(local_query_dir, exist_ok=True)
                    local_query_path = os.path.join(local_query_dir, f"view_{current_ulid}.sql")
                    
                    logger.info(f"Writing executed SQL to: {local_query_path}")
                    with open(local_query_path, 'w') as f:
                        f.write(retrieved_sql)
                        
                    # Cleanup original input file if applicable
                    if is_input_file and os.path.exists(sql_input):
                        logger.info(f"Removing original input file: {sql_input}")
                        os.remove(sql_input)
                else:
                    logger.warning("Could not extract SQL from downloaded artifact.")
            except Exception as ae:
                logger.error(f"Failed to retrieve/process execution artifact: {ae}")

            logger.info("Checking for table to profile...")
            target_table = extract_table_name(sql_content)
            
            if target_table and profile_table:
                logger.info(f"Identified target table: {target_table}. Initiating profiling...")
                
                # Define output configuration
                output_dir = "logs/pipeline/uc_sql_view_fill_rates/"
                filename = f"profile_{current_ulid}.json"
                
                try:
                    profile_table(
                        target_table, 
                        output_dir=output_dir, 
                        profile=PROFILE, 
                        custom_filename=filename
                    )
                    logger.info(f"Profiling completed. Saved to {output_dir}{filename}")
                    
                    if calculate_fill_rate:
                        schema_path = f"{output_dir}{filename}"
                        fillrate_filename = f"fillrate_{current_ulid}.json"
                        fillrate_path = f"{output_dir}{fillrate_filename}"
                        
                        logger.info(f"Initiating fill rate calculation...")
                        try:
                            calculate_fill_rate(
                                schema_path=schema_path,
                                output_path=fillrate_path,
                                profile=PROFILE
                            )
                            logger.info(f"Fill rate calculation completed. Saved to {fillrate_path}")
                        except Exception as fe:
                             logger.error(f"Fill rate calculation failed: {fe}")

                except Exception as pe:
                    logger.error(f"Profiling failed: {pe}")
            elif not target_table:
                logger.info("No target table (catalog.schema.table) found in SQL to profile.")
            else:
                logger.warning("Target table found but profiling module is not available.")
            
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

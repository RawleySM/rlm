#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import json
import subprocess
import sys
import os
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PROFILE = "rstanhope"
DEFAULT_WAREHOUSE_ID = "57f6389fdcdefbc0"  # spendmend-dev-sql-cluster
DEFAULT_PRECISION = 4
TABLE_TIMEOUT_SECONDS = 180
DEFAULT_HTTP_TIMEOUT_SECONDS = 900
FAIL_FLAG = "FAIL"
DEFAULT_WAIT_TIMEOUT_SECONDS = 50
DEFAULT_WAIT_TIMEOUT = f"{DEFAULT_WAIT_TIMEOUT_SECONDS}s"

@dataclass
class CommandResult:
    stdout: str | None
    timed_out: bool = False

@dataclass
class FillRateResult:
    rowcount: int | None
    rates: list[list[str, float]] | None
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.rowcount is not None and self.rates is not None and not self.timed_out

def run_databricks_cmd(args, description, profile_name=DEFAULT_PROFILE, timeout=None, http_timeout=None, quiet=False):
    """Runs a databricks CLI command and returns the stdout."""
    if not quiet:
        print(f"Running: {description}...")
    env = os.environ.copy()
    if http_timeout:
        env["DATABRICKS_HTTP_TIMEOUT"] = str(http_timeout)
        env["DATABRICKS_HTTP_TIMEOUT_SECONDS"] = str(http_timeout)

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
            env=env,
        )
        return CommandResult(result.stdout)
    except subprocess.TimeoutExpired:
        if not quiet:
            print(f"Command timed out after {timeout}s: {' '.join(args)}")
        return CommandResult(None, timed_out=True)
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.lower() if e.stderr else ""
        if "invalid access token" in err_msg or "unauthenticated" in err_msg:
            print(f"\n[CRITICAL AUTH ERROR] Failed to authenticate with profile '{profile_name}'.")
            print(f"Details: {e.stderr.strip()}")
            print(f"Please run 'databricks auth login --profile {profile_name}' to refresh your credentials.\n")
            sys.exit(1)

        print(f"Error running command: {' '.join(args)}")
        print(f"Stderr: {e.stderr}")
        return CommandResult(None)

def escape_identifier(name: str) -> str:
    return name.replace("`", "``") if name else name

def extract_column_names_from_schema(data: dict):
    columns = data.get("columns") or []
    return [col["name"] for col in columns if isinstance(col, dict) and "name" in col]

def execute_inline_statement(statement, catalog, schema, warehouse_id, profile, wait_timeout, table_timeout, http_timeout):
    payload = {
        "statement": statement,
        "warehouse_id": warehouse_id,
        "catalog": catalog,
        "schema": schema,
        "disposition": "INLINE",
        "wait_timeout": "0s",
    }
    cmd = [
        "databricks",
        "api",
        "post",
        "/api/2.0/sql/statements/",
        "--output",
        "json",
        "--json",
        json.dumps(payload),
    ]
    if profile:
        cmd.extend(["--profile", profile])

    output = run_databricks_cmd(
        cmd,
        f"Submitting SQL in {catalog}.{schema}",
        profile,
        timeout=30,
        http_timeout=http_timeout,
    )
    if output.timed_out:
        return None, True

    if not output.stdout:
        return None, False

    try:
        resp = json.loads(output.stdout)
    except json.JSONDecodeError:
        print("Failed to parse SQL execution response.")
        return None, False

    statement_id = resp.get("statement_id")
    state = (resp.get("status", {}) or {}).get("state", "UNKNOWN").upper()

    start_time = time.time()
    while state not in {"SUCCEEDED", "FINISHED"} and state not in {"FAILED", "CANCELED", "CLOSED"}:
        if not statement_id:
            break
        if time.time() - start_time > table_timeout:
            print(f"Polling timed out after {table_timeout}s")
            return None, True

        time.sleep(2)

        poll_cmd = ["databricks", "api", "get", f"/api/2.0/sql/statements/{statement_id}", "--output", "json"]
        if profile:
            poll_cmd.extend(["--profile", profile])

        poll_out = run_databricks_cmd(poll_cmd, "Polling...", profile, timeout=10, quiet=True)
        if poll_out.stdout:
            try:
                resp = json.loads(poll_out.stdout)
                state = (resp.get("status", {}) or {}).get("state", "UNKNOWN").upper()
            except json.JSONDecodeError:
                pass

    if state not in {"SUCCEEDED", "FINISHED"}:
        print(f"SQL execution failed or timed out (state={state}).")
        return None, False

    result = resp.get("result") or {}
    rows = result.get("data_array") or result.get("data") or []
    manifest_cols = resp.get("manifest", {}).get("schema", {}).get("columns", [])
    col_names = [col.get("name") for col in manifest_cols if col.get("name")]

    if not rows or not col_names:
        print("SQL execution returned no data.")
        return None, False

    first_row = rows[0]
    if len(first_row) != len(col_names):
        print("SQL execution returned unexpected column count.")
        return None, False

    return dict(zip(col_names, first_row)), False

def compute_fill_rates(
    catalog,
    schema,
    table,
    columns,
    warehouse_id,
    profile,
    precision,
    wait_timeout,
    table_timeout,
    http_timeout,
):
    select_parts = ["COUNT(*) AS __total"]
    for col in columns:
        safe_col = escape_identifier(col)
        select_parts.append(f"COUNT(`{safe_col}`) AS `{safe_col}`")

    statement = (
        f"SELECT {', '.join(select_parts)} "
        f"FROM `{escape_identifier(catalog)}`.`{escape_identifier(schema)}`.`{escape_identifier(table)}`"
    )
    row, timed_out = execute_inline_statement(
        statement, catalog, schema, warehouse_id, profile, wait_timeout, table_timeout, http_timeout
    )
    if timed_out:
        return FillRateResult(None, None, True)
    if not row:
        return FillRateResult(None, None, False)

    def _to_num(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            pass
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    total = _to_num(row.pop("__total", 0))
    rowcount = int(total) if isinstance(total, (int, float)) else 0
    if not total:
        return FillRateResult(rowcount, [[col, 0] for col in columns])

    fill_rates = []
    for col in columns:
        non_nulls = _to_num(row.get(col) or row.get(escape_identifier(col)) or 0)
        rate = round(non_nulls / total, precision)
        fill_rates.append([col, rate])
    return FillRateResult(rowcount, fill_rates)

def save_fill_rate_json(name, rowcount, rates, dest: Path | None):

    data = {name: [rowcount, rates]}

    if dest:

        with dest.open("w", encoding="utf-8") as fh:

            json.dump(data, fh, indent=2)

        print(f"Saved fill rates for {name} to {dest}")

    else:

        print(json.dumps(data, indent=2))



def calculate_fill_rate(schema_path: str, output_path: str, profile: str = DEFAULT_PROFILE, warehouse_id: str = DEFAULT_WAREHOUSE_ID):

    """

    Calculates fill rate for a table defined in a schema file and saves to output path as JSON.

    """

    path = Path(schema_path)

    if not path.exists():

        print(f"Schema file not found: {path}")

        return



    try:

        data = json.loads(path.read_text(encoding="utf-8"))

    except Exception as exc:

        print(f"Failed to parse JSON from {path}: {exc}")

        return



    # Extract catalog, schema, table

    full_name = data.get("full_name")

    catalog = data.get("catalog_name")

    schema = data.get("schema_name")

    table = data.get("name")

    

    if not (catalog and schema and table):

         # Try to parse from full_name if available

         if full_name:

             parts = full_name.split(".")

             if len(parts) == 3:

                 catalog, schema, table = parts

    

    if not (catalog and schema and table):

        print(f"Missing catalog/schema/table in schema file: {path}")

        return



    if not full_name:

        full_name = f"{catalog}.{schema}.{table}"



    columns = extract_column_names_from_schema(data)

    if not columns:

        print(f"No columns found in schema file: {path}")

        return



    print(f"Calculating fill rates for {full_name}...")

    

    result = compute_fill_rates(

        catalog, 

        schema, 

        table, 

        columns, 

        warehouse_id, 

        profile, 

        DEFAULT_PRECISION, 

        DEFAULT_WAIT_TIMEOUT, 

        TABLE_TIMEOUT_SECONDS, 

        DEFAULT_HTTP_TIMEOUT_SECONDS

    )



    if result.success:

        save_fill_rate_json(full_name, result.rowcount, result.rates, Path(output_path))

    else:

        print(f"Failed to calculate fill rates for {full_name}")
#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
import argparse
import subprocess
import json
import time
import sys
import os
from pathlib import Path

DEFAULT_PROFILE = "rstanhope"

def run_databricks_cmd(args, description):
    """Runs a databricks CLI command and returns the stdout."""
    print(f"Running: {description}...")
    try:
        # Capture output
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.lower() if e.stderr else ""
        if "invalid access token" in err_msg or "unauthenticated" in err_msg:
             print(f"\n[CRITICAL AUTH ERROR] Failed to authenticate with profile '{DEFAULT_PROFILE}'.")
             print(f"Details: {e.stderr.strip()}")
             print(f"Please run 'databricks auth login --profile {DEFAULT_PROFILE}' to refresh your credentials.\n")
             sys.exit(1)
        
        print(f"Error running command: {' '.join(args)}")
        print(f"Stderr: {e.stderr}")
        return None

def list_schemas(catalog, profile):
    """Lists schemas in a catalog."""
    cmd = ["databricks", "schemas", "list", catalog, "--output", "json"]
    if profile:
        cmd.extend(["--profile", profile])
    
    output = run_databricks_cmd(cmd, f"Listing schemas in {catalog}")
    if not output:
        return []
    
    try:
        data = json.loads(output)
        # Handle list of dicts or dict with 'schemas' key
        if isinstance(data, list):
             return [item['name'] for item in data if 'name' in item]
        elif isinstance(data, dict):
             # Try to find a list value that looks like schemas, or specific keys
             if 'schemas' in data:
                 return [item['name'] for item in data['schemas']]
             # Fallback: check if it returns a single object
             if 'name' in data:
                 return [data['name']]
        return [] 
    except json.JSONDecodeError:
        print("Failed to decode JSON from schemas list")
        return []

def list_tables(catalog, schema, profile):
    """Lists tables in a catalog.schema."""
    cmd = ["databricks", "tables", "list", catalog, schema, "--output", "json"]
    if profile:
        cmd.extend(["--profile", profile])

    output = run_databricks_cmd(cmd, f"Listing tables in {catalog}.{schema}")
    if not output:
        return []

    try:
        data = json.loads(output)
        tables = []
        
        # Handle potential API response variations
        # Sometimes lists return just a list of objects
        if isinstance(data, list):
            items = data
        else:
            # Or a wrapper dict
            items = data.get('tables', [])
            if not items and 'name' in data:
                 # Single item return?
                 items = [data]
        
        for item in items:
            if 'name' in item:
                tables.append(item['name'])
        return tables
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from tables list for {schema}")
        return []

def get_table_details(full_table_name, profile):
    """Gets details for a specific table."""
    cmd = ["databricks", "tables", "get", full_table_name, "--output", "json"]
    if profile:
        cmd.extend(["--profile", profile])
    
    return run_databricks_cmd(cmd, f"Getting details for {full_table_name}")

def save_schema(catalog, schema, table_name, content, base_dir="app_data", custom_filename=None):
    """Saves the schema JSON to file."""
    if not content:
        return

    if custom_filename:
        # Save directly to base_dir with custom filename
        path = Path(base_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / custom_filename
    else:
        # Create directory: app_data/<catalog>/<schema>/
        path = Path(base_dir) / catalog / schema
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{table_name}_schema.json"
    
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Saved: {file_path}")
    except Exception as e:
        print(f"Failed to write file {file_path}: {e}")

def profile_table(full_table_name, output_dir="app_data", profile=DEFAULT_PROFILE, custom_filename=None):
    """
    Profiles a single table by fetching its details and saving the schema.
    
    Args:
        full_table_name (str): The full table name (catalog.schema.table).
        output_dir (str): Base directory for output.
        profile (str): Databricks CLI profile to use.
        custom_filename (str): Optional custom filename for the output.
    """
    parts = full_table_name.split('.')
    if len(parts) != 3:
        print(f"Invalid table name format: {full_table_name}. Expected catalog.schema.table")
        return

    catalog, schema, table = parts
    print(f"Profiling table: {full_table_name}...")
    content = get_table_details(full_table_name, profile)
    if content:
        save_schema(catalog, schema, table, content, output_dir, custom_filename)
    else:
        print(f"Failed to retrieve details for {full_table_name}")

def main():
    parser = argparse.ArgumentParser(description="Bulk profile Databricks tables.")
    parser.add_argument("target", help="Target catalog, catalog__schema, or full table name (catalog.schema.table)")
    parser.add_argument("--output-dir", default="app_data", help="Base directory for output (default: app_data)")
    
    args = parser.parse_args()

    target = args.target
    profile = DEFAULT_PROFILE
    
    # Check if target is a specific table (3 components separated by dots)
    if target.count('.') == 2:
        profile_table(target, args.output_dir, profile)
        return

    if "__" in target:
        catalog, schema = target.split("__", 1)
        schemas_to_process = [schema]
        print(f"Targeting specific schema: {catalog}.{schema}")
    else:
        catalog = target
        print(f"Fetching schemas for catalog: {catalog}")
        schemas_to_process = list_schemas(catalog, profile)
        print(f"Found {len(schemas_to_process)} schemas: {schemas_to_process}")

    for schema in schemas_to_process:
        print(f"\n--- Processing Schema: {schema} ---")
        tables = list_tables(catalog, schema, profile)
        print(f"Found {len(tables)} tables in {catalog}.{schema}")
        
        for table in tables:
            full_table_name = f"{catalog}.{schema}.{table}"
            profile_table(full_table_name, args.output_dir, profile)
            
            print("Waiting 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()

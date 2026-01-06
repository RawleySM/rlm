"""Unity Catalog tools for managing catalogs, schemas, tables, and volumes.

Provides tools for exploring and operating on Unity Catalog resources
including hospital chain ERP databases stored in separate volumes.

Supports dual-mode execution:
- Local mode: Uses REST API calls to Databricks workspace
- Native mode: Uses SparkSession SQL directly on Databricks cluster
"""

import os

from google.adk.tools import ToolContext

from rlm_adk.runtime import get_execution_mode, get_spark_session


def list_catalogs(tool_context: ToolContext) -> dict:
    """List all available Unity Catalogs in the Databricks workspace.

    Use this tool to discover which catalogs are available, including
    hospital chain ERP databases and the masterdata vendor catalog.

    Supports dual-mode execution:
    - Native mode: Uses SparkSession SHOW CATALOGS
    - Local mode: Uses REST API calls

    Returns:
        dict: Contains catalog listing results.
            - 'status' (str): "success" or "error"
            - 'catalogs' (list, optional): List of catalog info dicts with
              'name', 'owner', 'comment', and 'created_at'
            - 'count' (int, optional): Number of catalogs found
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        # Native mode: use SparkSession directly
        if get_execution_mode() == "native":
            return _native_list_catalogs()

        # Local mode: use REST API
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not all([databricks_host, databricks_token]):
            return _simulate_list_catalogs()

        return _api_list_catalogs(databricks_host, databricks_token)

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to list catalogs: {str(e)}",
        }


def list_schemas(catalog_name: str, tool_context: ToolContext) -> dict:
    """List all schemas within a specific Unity Catalog.

    Use this tool to explore the database schemas within a hospital chain's
    ERP catalog or the masterdata vendor catalog.

    Supports dual-mode execution:
    - Native mode: Uses SparkSession SHOW SCHEMAS
    - Local mode: Uses REST API calls

    Args:
        catalog_name: The name of the catalog to list schemas from.

    Returns:
        dict: Contains schema listing results.
            - 'status' (str): "success" or "error"
            - 'catalog' (str): The catalog that was queried
            - 'schemas' (list, optional): List of schema info dicts with
              'name', 'owner', 'comment'
            - 'count' (int, optional): Number of schemas found
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        # Native mode: use SparkSession directly
        if get_execution_mode() == "native":
            return _native_list_schemas(catalog_name)

        # Local mode: use REST API
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not all([databricks_host, databricks_token]):
            return _simulate_list_schemas(catalog_name)

        return _api_list_schemas(databricks_host, databricks_token, catalog_name)

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to list schemas: {str(e)}",
        }


def list_tables(
    catalog_name: str, schema_name: str, tool_context: ToolContext
) -> dict:
    """List all tables and views within a specific schema.

    Use this tool to discover available tables in a hospital's ERP schema
    or vendor data schema, including their types (TABLE, VIEW, MATERIALIZED VIEW).

    Supports dual-mode execution:
    - Native mode: Uses SparkSession SHOW TABLES
    - Local mode: Uses REST API calls

    Args:
        catalog_name: The name of the catalog containing the schema.
        schema_name: The name of the schema to list tables from.

    Returns:
        dict: Contains table listing results.
            - 'status' (str): "success" or "error"
            - 'catalog' (str): The catalog that was queried
            - 'schema' (str): The schema that was queried
            - 'tables' (list, optional): List of table info dicts with
              'name', 'type', 'owner', 'comment', 'created_at'
            - 'count' (int, optional): Number of tables found
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        # Native mode: use SparkSession directly
        if get_execution_mode() == "native":
            return _native_list_tables(catalog_name, schema_name)

        # Local mode: use REST API
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not all([databricks_host, databricks_token]):
            return _simulate_list_tables(catalog_name, schema_name)

        return _api_list_tables(databricks_host, databricks_token, catalog_name, schema_name)

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to list tables: {str(e)}",
        }


def list_volumes(
    catalog_name: str, schema_name: str, tool_context: ToolContext
) -> dict:
    """List all volumes within a specific schema.

    Use this tool to discover Unity Catalog volumes containing raw files,
    data exports, or unstructured data from hospital ERP systems.

    Note: Both native and local modes use REST API for this operation,
    as Spark SQL does not have a SHOW VOLUMES command.

    Args:
        catalog_name: The name of the catalog containing the schema.
        schema_name: The name of the schema to list volumes from.

    Returns:
        dict: Contains volume listing results.
            - 'status' (str): "success" or "error"
            - 'catalog' (str): The catalog that was queried
            - 'schema' (str): The schema that was queried
            - 'volumes' (list, optional): List of volume info dicts with
              'name', 'volume_type', 'storage_location', 'owner', 'comment'
            - 'count' (int, optional): Number of volumes found
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        # Note: Both native and local modes use REST API for volumes
        # because Spark SQL doesn't have SHOW VOLUMES command
        if not all([databricks_host, databricks_token]):
            return _simulate_list_volumes(catalog_name, schema_name)

        # Native mode with credentials available: use REST API via native helper
        if get_execution_mode() == "native":
            return _native_list_volumes(
                databricks_host, databricks_token, catalog_name, schema_name
            )

        # Local mode: use REST API directly
        return _api_list_volumes(databricks_host, databricks_token, catalog_name, schema_name)

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to list volumes: {str(e)}",
        }


def get_volume_metadata(
    catalog_name: str,
    schema_name: str,
    volume_name: str,
    tool_context: ToolContext
) -> dict:
    """Get detailed metadata about a specific Unity Catalog volume.

    Use this tool to inspect volume properties, storage location, and
    access permissions for a hospital ERP data volume.

    Supports dual-mode execution:
    - Native mode: Uses SparkSession DESCRIBE VOLUME
    - Local mode: Uses REST API calls

    Args:
        catalog_name: The name of the catalog containing the volume.
        schema_name: The name of the schema containing the volume.
        volume_name: The name of the volume to get metadata for.

    Returns:
        dict: Contains volume metadata.
            - 'status' (str): "success" or "error"
            - 'volume' (dict, optional): Volume details including
              'full_name', 'volume_type', 'storage_location', 'owner',
              'created_at', 'updated_at', 'comment'
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        # Native mode: use SparkSession DESCRIBE VOLUME
        if get_execution_mode() == "native":
            return _native_get_volume_metadata(catalog_name, schema_name, volume_name)

        # Local mode: use REST API
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not all([databricks_host, databricks_token]):
            return _simulate_get_volume_metadata(catalog_name, schema_name, volume_name)

        return _api_get_volume_metadata(
            databricks_host, databricks_token, catalog_name, schema_name, volume_name
        )

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to get volume metadata: {str(e)}",
        }


def read_table_sample(
    catalog_name: str,
    schema_name: str,
    table_name: str,
    sample_size: int,
    tool_context: ToolContext
) -> dict:
    """Read a sample of rows from a table for data exploration.

    Use this tool to preview data in a hospital ERP table or vendor table
    to understand its structure and content before performing analysis.

    Args:
        catalog_name: The name of the catalog containing the table.
        schema_name: The name of the schema containing the table.
        table_name: The name of the table to sample from.
        sample_size: Number of rows to retrieve (max 1000).

    Returns:
        dict: Contains sample data results.
            - 'status' (str): "success" or "error"
            - 'table' (str): Full table name (catalog.schema.table)
            - 'columns' (list, optional): List of column info with
              'name', 'type', 'nullable'
            - 'data' (list, optional): Sample rows as list of dicts
            - 'row_count' (int, optional): Number of rows returned
            - 'total_rows' (int, optional): Total rows in table (if available)
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        sample_size = min(sample_size, 1000)

        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not all([databricks_host, databricks_token]):
            return _simulate_read_table_sample(
                catalog_name, schema_name, table_name, sample_size
            )

        # Use SQL execution API
        from rlm_adk.tools.databricks_repl import execute_sql_query

        full_name = f"{catalog_name}.{schema_name}.{table_name}"
        query = f"SELECT * FROM {full_name} LIMIT {sample_size}"

        result = execute_sql_query(query, tool_context)

        if result["status"] == "success":
            return {
                "status": "success",
                "table": full_name,
                "columns": result.get("columns", []),
                "data": result.get("data", []),
                "row_count": result.get("row_count", 0),
            }
        return result

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to read table sample: {str(e)}",
        }


def create_view(
    catalog_name: str,
    schema_name: str,
    view_name: str,
    view_definition: str,
    replace_if_exists: bool,
    tool_context: ToolContext
) -> dict:
    """Create a view in Unity Catalog from a SQL definition.

    Use this tool to create analytical views that join data across
    hospital ERP databases or create unified vendor views.

    Args:
        catalog_name: The catalog where the view will be created.
        schema_name: The schema where the view will be created.
        view_name: The name for the new view.
        view_definition: The SQL SELECT statement defining the view.
        replace_if_exists: Whether to replace an existing view with same name.

    Returns:
        dict: Contains view creation results.
            - 'status' (str): "success" or "error"
            - 'view' (str, optional): Full view name (catalog.schema.view)
            - 'message' (str, optional): Success message
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        full_name = f"{catalog_name}.{schema_name}.{view_name}"

        if not all([databricks_host, databricks_token]):
            return {
                "status": "success",
                "view": full_name,
                "message": f"View {full_name} created successfully (simulated)",
            }

        # Use SQL execution API
        from rlm_adk.tools.databricks_repl import execute_sql_query

        create_stmt = "CREATE OR REPLACE VIEW" if replace_if_exists else "CREATE VIEW"
        query = f"{create_stmt} {full_name} AS {view_definition}"

        result = execute_sql_query(query, tool_context)

        if result["status"] == "success":
            return {
                "status": "success",
                "view": full_name,
                "message": f"View {full_name} created successfully",
            }
        return result

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to create view: {str(e)}",
        }


# Native implementation functions (SparkSession on Databricks cluster)


def _native_list_catalogs() -> dict:
    """List catalogs using SparkSession in native mode."""
    spark = get_spark_session()
    if spark is None:
        raise RuntimeError("SparkSession not available in native mode")

    rows = spark.sql("SHOW CATALOGS").collect()
    return {
        "status": "success",
        "catalogs": [{"name": row.catalog} for row in rows],
        "count": len(rows),
    }


def _native_list_schemas(catalog_name: str) -> dict:
    """List schemas using SparkSession in native mode."""
    spark = get_spark_session()
    if spark is None:
        raise RuntimeError("SparkSession not available in native mode")

    rows = spark.sql(f"SHOW SCHEMAS IN {catalog_name}").collect()
    # SHOW SCHEMAS returns 'databaseName' or 'namespace' column depending on version
    schemas = []
    for row in rows:
        # Try different possible column names
        name = None
        if hasattr(row, "databaseName"):
            name = row.databaseName
        elif hasattr(row, "namespace"):
            name = row.namespace
        elif hasattr(row, "schemaName"):
            name = row.schemaName
        else:
            # Fallback to first column
            name = row[0]
        schemas.append({"name": name, "owner": None, "comment": None})

    return {
        "status": "success",
        "catalog": catalog_name,
        "schemas": schemas,
        "count": len(schemas),
    }


def _native_list_tables(catalog_name: str, schema_name: str) -> dict:
    """List tables using SparkSession in native mode."""
    spark = get_spark_session()
    if spark is None:
        raise RuntimeError("SparkSession not available in native mode")

    rows = spark.sql(f"SHOW TABLES IN {catalog_name}.{schema_name}").collect()
    tables = []
    for row in rows:
        # SHOW TABLES returns columns: database, tableName, isTemporary
        name = None
        is_temporary = False
        if hasattr(row, "tableName"):
            name = row.tableName
        elif hasattr(row, "table"):
            name = row.table
        else:
            # Fallback to second column (first is database)
            name = row[1] if len(row) > 1 else row[0]

        if hasattr(row, "isTemporary"):
            is_temporary = row.isTemporary

        tables.append({
            "name": name,
            "type": "TEMPORARY" if is_temporary else "TABLE",
            "owner": None,
            "comment": None,
            "created_at": None,
        })

    return {
        "status": "success",
        "catalog": catalog_name,
        "schema": schema_name,
        "tables": tables,
        "count": len(tables),
    }


def _native_list_volumes(
    host: str, token: str, catalog_name: str, schema_name: str
) -> dict:
    """List volumes - falls back to REST API as Spark SQL lacks SHOW VOLUMES.

    Note: Spark SQL doesn't have a SHOW VOLUMES command, so even in native mode
    we use the REST API for this operation.
    """
    # Fall back to REST API as Spark SQL doesn't support SHOW VOLUMES
    return _api_list_volumes(host, token, catalog_name, schema_name)


def _native_get_volume_metadata(
    catalog_name: str, schema_name: str, volume_name: str
) -> dict:
    """Get volume metadata using SparkSession DESCRIBE VOLUME in native mode."""
    spark = get_spark_session()
    if spark is None:
        raise RuntimeError("SparkSession not available in native mode")

    full_name = f"{catalog_name}.{schema_name}.{volume_name}"
    rows = spark.sql(f"DESCRIBE VOLUME {full_name}").collect()

    # DESCRIBE VOLUME returns key-value pairs
    volume_info = {
        "full_name": full_name,
        "volume_type": None,
        "storage_location": None,
        "owner": None,
        "created_at": None,
        "updated_at": None,
        "comment": None,
    }

    for row in rows:
        # Rows typically have 'col_name' and 'data_type' or 'info_name' and 'info_value'
        key = None
        value = None
        if hasattr(row, "info_name"):
            key = row.info_name.lower() if row.info_name else None
            value = row.info_value
        elif hasattr(row, "col_name"):
            key = row.col_name.lower() if row.col_name else None
            value = row.data_type
        elif len(row) >= 2:
            key = str(row[0]).lower() if row[0] else None
            value = row[1]

        if key:
            if "type" in key and "volume" in key:
                volume_info["volume_type"] = value
            elif "location" in key or "storage" in key:
                volume_info["storage_location"] = value
            elif key == "owner":
                volume_info["owner"] = value
            elif "created" in key:
                volume_info["created_at"] = value
            elif "updated" in key or "modified" in key:
                volume_info["updated_at"] = value
            elif key == "comment":
                volume_info["comment"] = value
            elif key == "name":
                volume_info["full_name"] = value

    return {
        "status": "success",
        "volume": volume_info,
    }


# Simulation functions for development/testing


def _simulate_list_catalogs() -> dict:
    """Simulate catalog listing for development."""
    return {
        "status": "success",
        "catalogs": [
            {
                "name": "healthcare_main",
                "owner": "admin",
                "comment": "Main healthcare analytics catalog",
                "created_at": "2024-01-15T10:00:00Z",
            },
            {
                "name": "hospital_chain_alpha",
                "owner": "data_team",
                "comment": "Alpha Hospital Chain ERP data",
                "created_at": "2024-02-01T08:00:00Z",
            },
            {
                "name": "hospital_chain_beta",
                "owner": "data_team",
                "comment": "Beta Hospital Network ERP data",
                "created_at": "2024-02-15T09:00:00Z",
            },
            {
                "name": "hospital_chain_gamma",
                "owner": "data_team",
                "comment": "Gamma Health System ERP data",
                "created_at": "2024-03-01T07:00:00Z",
            },
            {
                "name": "masterdata_vendors",
                "owner": "mdm_team",
                "comment": "Master data management - Vendor golden records",
                "created_at": "2024-01-01T06:00:00Z",
            },
        ],
        "count": 5,
    }


def _simulate_list_schemas(catalog_name: str) -> dict:
    """Simulate schema listing for development."""
    schemas_by_catalog = {
        "hospital_chain_alpha": [
            {"name": "erp_vendors", "owner": "data_team", "comment": "Vendor master data"},
            {"name": "erp_transactions", "owner": "data_team", "comment": "Purchase transactions"},
            {"name": "erp_inventory", "owner": "data_team", "comment": "Inventory data"},
        ],
        "hospital_chain_beta": [
            {"name": "vendor_data", "owner": "data_team", "comment": "Vendor information"},
            {"name": "purchasing", "owner": "data_team", "comment": "Purchasing records"},
            {"name": "supplies", "owner": "data_team", "comment": "Supply management"},
        ],
        "hospital_chain_gamma": [
            {"name": "suppliers", "owner": "data_team", "comment": "Supplier database"},
            {"name": "orders", "owner": "data_team", "comment": "Order history"},
            {"name": "materials", "owner": "data_team", "comment": "Materials catalog"},
        ],
        "masterdata_vendors": [
            {"name": "golden_records", "owner": "mdm_team", "comment": "Verified vendor entities"},
            {"name": "mappings", "owner": "mdm_team", "comment": "Source to golden record mappings"},
            {"name": "audit", "owner": "mdm_team", "comment": "Audit trail for changes"},
        ],
    }

    schemas = schemas_by_catalog.get(
        catalog_name,
        [{"name": "default", "owner": "admin", "comment": "Default schema"}],
    )

    return {
        "status": "success",
        "catalog": catalog_name,
        "schemas": schemas,
        "count": len(schemas),
    }


def _simulate_list_tables(catalog_name: str, schema_name: str) -> dict:
    """Simulate table listing for development."""
    tables = [
        {
            "name": "vendors",
            "type": "TABLE",
            "owner": "data_team",
            "comment": "Vendor master records",
            "created_at": "2024-01-20T10:00:00Z",
        },
        {
            "name": "vendor_addresses",
            "type": "TABLE",
            "owner": "data_team",
            "comment": "Vendor address information",
            "created_at": "2024-01-20T10:05:00Z",
        },
        {
            "name": "vendor_contacts",
            "type": "TABLE",
            "owner": "data_team",
            "comment": "Vendor contact details",
            "created_at": "2024-01-20T10:10:00Z",
        },
        {
            "name": "active_vendors_view",
            "type": "VIEW",
            "owner": "data_team",
            "comment": "Active vendors only",
            "created_at": "2024-02-01T08:00:00Z",
        },
    ]

    return {
        "status": "success",
        "catalog": catalog_name,
        "schema": schema_name,
        "tables": tables,
        "count": len(tables),
    }


def _simulate_list_volumes(catalog_name: str, schema_name: str) -> dict:
    """Simulate volume listing for development."""
    volumes = [
        {
            "name": "raw_exports",
            "volume_type": "MANAGED",
            "storage_location": f"s3://databricks-data/{catalog_name}/{schema_name}/raw_exports",
            "owner": "data_team",
            "comment": "Raw ERP export files",
        },
        {
            "name": "vendor_documents",
            "volume_type": "MANAGED",
            "storage_location": f"s3://databricks-data/{catalog_name}/{schema_name}/vendor_documents",
            "owner": "data_team",
            "comment": "Vendor contracts and documents",
        },
    ]

    return {
        "status": "success",
        "catalog": catalog_name,
        "schema": schema_name,
        "volumes": volumes,
        "count": len(volumes),
    }


def _simulate_get_volume_metadata(
    catalog_name: str, schema_name: str, volume_name: str
) -> dict:
    """Simulate volume metadata retrieval."""
    full_name = f"{catalog_name}.{schema_name}.{volume_name}"

    return {
        "status": "success",
        "volume": {
            "full_name": full_name,
            "volume_type": "MANAGED",
            "storage_location": f"s3://databricks-data/{catalog_name}/{schema_name}/{volume_name}",
            "owner": "data_team",
            "created_at": "2024-01-20T10:00:00Z",
            "updated_at": "2024-06-01T15:30:00Z",
            "comment": f"Volume containing data for {schema_name}",
        },
    }


def _simulate_read_table_sample(
    catalog_name: str, schema_name: str, table_name: str, sample_size: int
) -> dict:
    """Simulate table sample read."""
    full_name = f"{catalog_name}.{schema_name}.{table_name}"

    # Generate sample vendor data
    sample_data = [
        {
            "vendor_id": "V001",
            "vendor_name": "MedSupply Corp",
            "tax_id": "12-3456789",
            "address": "123 Medical Dr, Boston, MA",
            "status": "ACTIVE",
        },
        {
            "vendor_id": "V002",
            "vendor_name": "HealthEquip Inc",
            "tax_id": "98-7654321",
            "address": "456 Health Ave, Chicago, IL",
            "status": "ACTIVE",
        },
        {
            "vendor_id": "V003",
            "vendor_name": "Surgical Solutions LLC",
            "tax_id": "55-1234567",
            "address": "789 Surgery Ln, Houston, TX",
            "status": "ACTIVE",
        },
    ][:sample_size]

    return {
        "status": "success",
        "table": full_name,
        "columns": [
            {"name": "vendor_id", "type": "STRING", "nullable": False},
            {"name": "vendor_name", "type": "STRING", "nullable": False},
            {"name": "tax_id", "type": "STRING", "nullable": True},
            {"name": "address", "type": "STRING", "nullable": True},
            {"name": "status", "type": "STRING", "nullable": False},
        ],
        "data": sample_data,
        "row_count": len(sample_data),
        "total_rows": 15847,
    }


# API implementation functions


def _api_list_catalogs(host: str, token: str) -> dict:
    """List catalogs via Unity Catalog API."""
    import requests

    url = f"https://{host}/api/2.1/unity-catalog/catalogs"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    catalogs = [
        {
            "name": c["name"],
            "owner": c.get("owner"),
            "comment": c.get("comment"),
            "created_at": c.get("created_at"),
        }
        for c in data.get("catalogs", [])
    ]

    return {"status": "success", "catalogs": catalogs, "count": len(catalogs)}


def _api_list_schemas(host: str, token: str, catalog_name: str) -> dict:
    """List schemas via Unity Catalog API."""
    import requests

    url = f"https://{host}/api/2.1/unity-catalog/schemas"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"catalog_name": catalog_name}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    schemas = [
        {
            "name": s["name"],
            "owner": s.get("owner"),
            "comment": s.get("comment"),
        }
        for s in data.get("schemas", [])
    ]

    return {
        "status": "success",
        "catalog": catalog_name,
        "schemas": schemas,
        "count": len(schemas),
    }


def _api_list_tables(
    host: str, token: str, catalog_name: str, schema_name: str
) -> dict:
    """List tables via Unity Catalog API."""
    import requests

    url = f"https://{host}/api/2.1/unity-catalog/tables"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"catalog_name": catalog_name, "schema_name": schema_name}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    tables = [
        {
            "name": t["name"],
            "type": t.get("table_type", "TABLE"),
            "owner": t.get("owner"),
            "comment": t.get("comment"),
            "created_at": t.get("created_at"),
        }
        for t in data.get("tables", [])
    ]

    return {
        "status": "success",
        "catalog": catalog_name,
        "schema": schema_name,
        "tables": tables,
        "count": len(tables),
    }


def _api_list_volumes(
    host: str, token: str, catalog_name: str, schema_name: str
) -> dict:
    """List volumes via Unity Catalog API."""
    import requests

    url = f"https://{host}/api/2.1/unity-catalog/volumes"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"catalog_name": catalog_name, "schema_name": schema_name}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    volumes = [
        {
            "name": v["name"],
            "volume_type": v.get("volume_type"),
            "storage_location": v.get("storage_location"),
            "owner": v.get("owner"),
            "comment": v.get("comment"),
        }
        for v in data.get("volumes", [])
    ]

    return {
        "status": "success",
        "catalog": catalog_name,
        "schema": schema_name,
        "volumes": volumes,
        "count": len(volumes),
    }


def _api_get_volume_metadata(
    host: str, token: str, catalog_name: str, schema_name: str, volume_name: str
) -> dict:
    """Get volume metadata via Unity Catalog API."""
    import requests

    full_name = f"{catalog_name}.{schema_name}.{volume_name}"
    url = f"https://{host}/api/2.1/unity-catalog/volumes/{full_name}"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    return {
        "status": "success",
        "volume": {
            "full_name": data.get("full_name"),
            "volume_type": data.get("volume_type"),
            "storage_location": data.get("storage_location"),
            "owner": data.get("owner"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "comment": data.get("comment"),
        },
    }

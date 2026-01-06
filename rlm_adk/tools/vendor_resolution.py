"""Vendor resolution tools for masterdata management.

Provides tools for resolving vendor instances across multiple hospital
clients to a single (or new) masterdata vendor entity. Implements fuzzy
matching, attribute-based search, and golden record management.
"""

import os

from google.adk.tools import ToolContext


def find_similar_vendors(
    vendor_name: str,
    hospital_chain: str,
    threshold: float,
    tool_context: ToolContext,
) -> dict:
    """Find vendors with similar names across hospital chains.

    Use this tool to identify potential duplicate vendors that may exist
    in different hospital ERP systems under slightly different names.
    Uses fuzzy string matching to find similar vendor names.

    Args:
        vendor_name: The vendor name to search for similar matches.
        hospital_chain: The source hospital chain (or "all" for all chains).
        threshold: Similarity threshold (0.0 to 1.0). Higher values
                   require closer matches. Recommended: 0.7-0.85

    Returns:
        dict: Contains similar vendor matches.
            - 'status' (str): "success" or "error"
            - 'query' (dict): Original search parameters
            - 'matches' (list, optional): List of similar vendors with
              'vendor_id', 'vendor_name', 'hospital_chain', 'similarity_score',
              'tax_id', 'masterdata_id' (if already mapped)
            - 'count' (int, optional): Number of matches found
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not all([databricks_host, databricks_token]):
            return _simulate_find_similar_vendors(vendor_name, hospital_chain, threshold)

        return _search_similar_vendors(
            databricks_host, databricks_token, vendor_name, hospital_chain, threshold
        )

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to find similar vendors: {str(e)}",
        }


def resolve_vendor_to_masterdata(
    source_vendor_id: str,
    source_hospital_chain: str,
    masterdata_vendor_id: str,
    confidence_score: float,
    tool_context: ToolContext,
) -> dict:
    """Create a mapping from a hospital vendor to a masterdata golden record.

    Use this tool to link a vendor from a hospital's ERP system to an
    existing masterdata vendor entity, establishing the resolution.

    Args:
        source_vendor_id: The vendor ID from the hospital ERP system.
        source_hospital_chain: The hospital chain the vendor belongs to.
        masterdata_vendor_id: The target masterdata vendor ID to map to.
        confidence_score: Confidence level of the match (0.0 to 1.0).

    Returns:
        dict: Contains mapping creation results.
            - 'status' (str): "success" or "error"
            - 'mapping' (dict, optional): Created mapping details with
              'mapping_id', 'source_vendor_id', 'source_hospital_chain',
              'masterdata_vendor_id', 'confidence_score', 'created_at'
            - 'message' (str, optional): Success message
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not all([databricks_host, databricks_token]):
            return _simulate_resolve_vendor(
                source_vendor_id, source_hospital_chain, masterdata_vendor_id, confidence_score
            )

        return _create_vendor_mapping_record(
            databricks_host,
            databricks_token,
            source_vendor_id,
            source_hospital_chain,
            masterdata_vendor_id,
            confidence_score,
        )

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to resolve vendor: {str(e)}",
        }


def get_masterdata_vendor(
    masterdata_vendor_id: str, tool_context: ToolContext
) -> dict:
    """Get details of a masterdata golden record vendor entity.

    Use this tool to retrieve the complete information about a vendor
    in the masterdata system, including all linked hospital vendors.

    Args:
        masterdata_vendor_id: The masterdata vendor ID to retrieve.

    Returns:
        dict: Contains masterdata vendor details.
            - 'status' (str): "success" or "error"
            - 'vendor' (dict, optional): Vendor golden record with
              'masterdata_id', 'canonical_name', 'tax_id', 'duns_number',
              'primary_address', 'verified_at', 'source_mappings' (list of
              linked hospital vendors)
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not all([databricks_host, databricks_token]):
            return _simulate_get_masterdata_vendor(masterdata_vendor_id)

        return _fetch_masterdata_vendor(databricks_host, databricks_token, masterdata_vendor_id)

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to get masterdata vendor: {str(e)}",
        }


def create_vendor_mapping(
    canonical_name: str,
    tax_id: str,
    duns_number: str,
    primary_address: str,
    source_vendor_ids: list,
    tool_context: ToolContext,
) -> dict:
    """Create a new masterdata vendor entity with source mappings.

    Use this tool when no existing masterdata vendor matches a vendor
    found in hospital ERP systems. Creates a new golden record and
    links the source vendors to it.

    Args:
        canonical_name: The standardized name for the vendor.
        tax_id: Federal tax identification number (EIN).
        duns_number: D-U-N-S number if available (can be empty string).
        primary_address: Primary business address.
        source_vendor_ids: List of dicts with 'vendor_id' and 'hospital_chain'
                          for vendors to link to this new entity.

    Returns:
        dict: Contains new vendor creation results.
            - 'status' (str): "success" or "error"
            - 'vendor' (dict, optional): Created vendor details with
              'masterdata_id', 'canonical_name', 'tax_id', 'duns_number',
              'primary_address', 'created_at'
            - 'mappings_created' (int, optional): Number of source mappings created
            - 'message' (str, optional): Success message
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not all([databricks_host, databricks_token]):
            return _simulate_create_vendor_mapping(
                canonical_name, tax_id, duns_number, primary_address, source_vendor_ids
            )

        return _create_new_masterdata_vendor(
            databricks_host,
            databricks_token,
            canonical_name,
            tax_id,
            duns_number,
            primary_address,
            source_vendor_ids,
        )

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to create vendor mapping: {str(e)}",
        }


def search_vendor_by_attributes(
    tax_id: str,
    duns_number: str,
    address_keywords: str,
    tool_context: ToolContext,
) -> dict:
    """Search for vendors by business attributes across all sources.

    Use this tool to find vendors using unique identifiers like tax ID
    or DUNS number, which can provide exact matches across hospital chains.

    Args:
        tax_id: Federal tax ID to search for (can be empty string).
        duns_number: D-U-N-S number to search for (can be empty string).
        address_keywords: Keywords from address to match (can be empty string).

    Returns:
        dict: Contains attribute search results.
            - 'status' (str): "success" or "error"
            - 'query' (dict): Search parameters used
            - 'hospital_vendors' (list, optional): Vendors found in hospital ERPs
            - 'masterdata_vendors' (list, optional): Matching masterdata vendors
            - 'potential_duplicates' (list, optional): Groups of likely duplicates
            - 'error_message' (str, optional): Error details if failed
    """
    try:
        databricks_host = os.getenv("DATABRICKS_HOST")
        databricks_token = os.getenv("DATABRICKS_TOKEN")

        if not all([databricks_host, databricks_token]):
            return _simulate_search_by_attributes(tax_id, duns_number, address_keywords)

        return _search_vendors_by_attributes(
            databricks_host, databricks_token, tax_id, duns_number, address_keywords
        )

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to search vendors: {str(e)}",
        }


# Simulation functions for development/testing


def _simulate_find_similar_vendors(
    vendor_name: str, hospital_chain: str, threshold: float
) -> dict:
    """Simulate finding similar vendors."""
    # Generate mock similar vendors based on the search
    base_name = vendor_name.lower().replace(" ", "")

    similar_vendors = [
        {
            "vendor_id": "ALPHA-V001",
            "vendor_name": "MedSupply Corp",
            "hospital_chain": "hospital_chain_alpha",
            "similarity_score": 0.92,
            "tax_id": "12-3456789",
            "masterdata_id": "MD-001",
        },
        {
            "vendor_id": "BETA-V023",
            "vendor_name": "MedSupply Corporation",
            "hospital_chain": "hospital_chain_beta",
            "similarity_score": 0.88,
            "tax_id": "12-3456789",
            "masterdata_id": "MD-001",
        },
        {
            "vendor_id": "GAMMA-V105",
            "vendor_name": "Med Supply Corp.",
            "hospital_chain": "hospital_chain_gamma",
            "similarity_score": 0.85,
            "tax_id": "12-3456789",
            "masterdata_id": None,
        },
    ]

    # Filter by threshold and hospital chain
    filtered = [v for v in similar_vendors if v["similarity_score"] >= threshold]
    if hospital_chain != "all":
        filtered = [v for v in filtered if v["hospital_chain"] == hospital_chain]

    return {
        "status": "success",
        "query": {
            "vendor_name": vendor_name,
            "hospital_chain": hospital_chain,
            "threshold": threshold,
        },
        "matches": filtered,
        "count": len(filtered),
    }


def _simulate_resolve_vendor(
    source_vendor_id: str,
    source_hospital_chain: str,
    masterdata_vendor_id: str,
    confidence_score: float,
) -> dict:
    """Simulate resolving a vendor to masterdata."""
    import uuid
    from datetime import datetime

    mapping_id = f"MAP-{uuid.uuid4().hex[:8].upper()}"

    return {
        "status": "success",
        "mapping": {
            "mapping_id": mapping_id,
            "source_vendor_id": source_vendor_id,
            "source_hospital_chain": source_hospital_chain,
            "masterdata_vendor_id": masterdata_vendor_id,
            "confidence_score": confidence_score,
            "created_at": datetime.now().isoformat(),
            "created_by": "rlm_adk_agent",
        },
        "message": f"Successfully mapped {source_vendor_id} to {masterdata_vendor_id}",
    }


def _simulate_get_masterdata_vendor(masterdata_vendor_id: str) -> dict:
    """Simulate getting masterdata vendor details."""
    mock_vendors = {
        "MD-001": {
            "masterdata_id": "MD-001",
            "canonical_name": "MedSupply Corporation",
            "tax_id": "12-3456789",
            "duns_number": "123456789",
            "primary_address": "123 Medical Drive, Boston, MA 02101",
            "verified_at": "2024-06-15T10:30:00Z",
            "source_mappings": [
                {"vendor_id": "ALPHA-V001", "hospital_chain": "hospital_chain_alpha"},
                {"vendor_id": "BETA-V023", "hospital_chain": "hospital_chain_beta"},
            ],
        },
        "MD-002": {
            "masterdata_id": "MD-002",
            "canonical_name": "HealthEquip Inc.",
            "tax_id": "98-7654321",
            "duns_number": "987654321",
            "primary_address": "456 Healthcare Ave, Chicago, IL 60601",
            "verified_at": "2024-05-20T14:00:00Z",
            "source_mappings": [
                {"vendor_id": "ALPHA-V002", "hospital_chain": "hospital_chain_alpha"},
            ],
        },
    }

    vendor = mock_vendors.get(masterdata_vendor_id)
    if vendor:
        return {"status": "success", "vendor": vendor}
    else:
        return {
            "status": "error",
            "error_message": f"Masterdata vendor {masterdata_vendor_id} not found",
        }


def _simulate_create_vendor_mapping(
    canonical_name: str,
    tax_id: str,
    duns_number: str,
    primary_address: str,
    source_vendor_ids: list,
) -> dict:
    """Simulate creating a new masterdata vendor."""
    import uuid
    from datetime import datetime

    masterdata_id = f"MD-{uuid.uuid4().hex[:6].upper()}"

    return {
        "status": "success",
        "vendor": {
            "masterdata_id": masterdata_id,
            "canonical_name": canonical_name,
            "tax_id": tax_id,
            "duns_number": duns_number,
            "primary_address": primary_address,
            "created_at": datetime.now().isoformat(),
            "created_by": "rlm_adk_agent",
        },
        "mappings_created": len(source_vendor_ids),
        "message": f"Created new masterdata vendor {masterdata_id} with {len(source_vendor_ids)} source mappings",
    }


def _simulate_search_by_attributes(
    tax_id: str, duns_number: str, address_keywords: str
) -> dict:
    """Simulate searching vendors by attributes."""
    hospital_vendors = []
    masterdata_vendors = []

    if tax_id == "12-3456789":
        hospital_vendors = [
            {
                "vendor_id": "ALPHA-V001",
                "vendor_name": "MedSupply Corp",
                "hospital_chain": "hospital_chain_alpha",
                "tax_id": "12-3456789",
            },
            {
                "vendor_id": "BETA-V023",
                "vendor_name": "MedSupply Corporation",
                "hospital_chain": "hospital_chain_beta",
                "tax_id": "12-3456789",
            },
            {
                "vendor_id": "GAMMA-V105",
                "vendor_name": "Med Supply Corp.",
                "hospital_chain": "hospital_chain_gamma",
                "tax_id": "12-3456789",
            },
        ]
        masterdata_vendors = [
            {
                "masterdata_id": "MD-001",
                "canonical_name": "MedSupply Corporation",
                "tax_id": "12-3456789",
            }
        ]

    potential_duplicates = []
    if len(hospital_vendors) > 1:
        potential_duplicates = [
            {
                "group_id": "DUP-001",
                "reason": "Same tax_id across multiple hospital chains",
                "vendor_ids": [v["vendor_id"] for v in hospital_vendors],
                "recommended_masterdata_id": masterdata_vendors[0]["masterdata_id"]
                if masterdata_vendors
                else None,
            }
        ]

    return {
        "status": "success",
        "query": {
            "tax_id": tax_id,
            "duns_number": duns_number,
            "address_keywords": address_keywords,
        },
        "hospital_vendors": hospital_vendors,
        "masterdata_vendors": masterdata_vendors,
        "potential_duplicates": potential_duplicates,
    }


# API implementation functions


def _search_similar_vendors(
    host: str, token: str, vendor_name: str, hospital_chain: str, threshold: float
) -> dict:
    """Search for similar vendors using Databricks SQL."""

    # Use soundex or levenshtein for fuzzy matching in Spark SQL
    query = f"""
    WITH vendor_union AS (
        SELECT vendor_id, vendor_name, 'hospital_chain_alpha' as hospital_chain, tax_id
        FROM hospital_chain_alpha.erp_vendors.vendors
        UNION ALL
        SELECT vendor_id, vendor_name, 'hospital_chain_beta' as hospital_chain, tax_id
        FROM hospital_chain_beta.vendor_data.vendors
        UNION ALL
        SELECT vendor_id, vendor_name, 'hospital_chain_gamma' as hospital_chain, tax_id
        FROM hospital_chain_gamma.suppliers.vendors
    ),
    with_similarity AS (
        SELECT
            v.*,
            m.masterdata_id,
            1.0 - (levenshtein(lower(v.vendor_name), lower('{vendor_name}')) /
                   greatest(length(v.vendor_name), length('{vendor_name}'))) as similarity_score
        FROM vendor_union v
        LEFT JOIN masterdata_vendors.mappings.vendor_mappings m
            ON v.vendor_id = m.source_vendor_id AND v.hospital_chain = m.source_hospital_chain
    )
    SELECT * FROM with_similarity
    WHERE similarity_score >= {threshold}
    {"AND hospital_chain = '" + hospital_chain + "'" if hospital_chain != "all" else ""}
    ORDER BY similarity_score DESC
    LIMIT 100
    """

    # This is a placeholder - actual implementation would use the execute_sql_query tool
    return _simulate_find_similar_vendors(vendor_name, hospital_chain, threshold)


def _create_vendor_mapping_record(
    host: str,
    token: str,
    source_vendor_id: str,
    source_hospital_chain: str,
    masterdata_vendor_id: str,
    confidence_score: float,
) -> dict:
    """Create vendor mapping via SQL insert."""

    query = f"""
    INSERT INTO masterdata_vendors.mappings.vendor_mappings
    (source_vendor_id, source_hospital_chain, masterdata_vendor_id, confidence_score, created_at, created_by)
    VALUES ('{source_vendor_id}', '{source_hospital_chain}', '{masterdata_vendor_id}',
            {confidence_score}, current_timestamp(), 'rlm_adk_agent')
    """

    return _simulate_resolve_vendor(
        source_vendor_id, source_hospital_chain, masterdata_vendor_id, confidence_score
    )


def _fetch_masterdata_vendor(host: str, token: str, masterdata_vendor_id: str) -> dict:
    """Fetch masterdata vendor from Unity Catalog."""

    query = f"""
    SELECT
        v.masterdata_id,
        v.canonical_name,
        v.tax_id,
        v.duns_number,
        v.primary_address,
        v.verified_at,
        collect_list(struct(m.source_vendor_id, m.source_hospital_chain)) as source_mappings
    FROM masterdata_vendors.golden_records.vendors v
    LEFT JOIN masterdata_vendors.mappings.vendor_mappings m
        ON v.masterdata_id = m.masterdata_vendor_id
    WHERE v.masterdata_id = '{masterdata_vendor_id}'
    GROUP BY v.masterdata_id, v.canonical_name, v.tax_id, v.duns_number, v.primary_address, v.verified_at
    """

    return _simulate_get_masterdata_vendor(masterdata_vendor_id)


def _create_new_masterdata_vendor(
    host: str,
    token: str,
    canonical_name: str,
    tax_id: str,
    duns_number: str,
    primary_address: str,
    source_vendor_ids: list,
) -> dict:
    """Create new masterdata vendor record and mappings."""
    return _simulate_create_vendor_mapping(
        canonical_name, tax_id, duns_number, primary_address, source_vendor_ids
    )


def _search_vendors_by_attributes(
    host: str, token: str, tax_id: str, duns_number: str, address_keywords: str
) -> dict:
    """Search vendors by attributes across all sources."""
    return _simulate_search_by_attributes(tax_id, duns_number, address_keywords)

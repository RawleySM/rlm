# RLM-ADK Architectural Overview

The `rlm_adk` package is an agentic framework built on `google-adk` that implements the Recursive Language Model (RLM) paradigm. It is specifically designed for healthcare data science tasks, such as vendor resolution across Databricks Unity Catalog volumes.

## High-Level Architecture

The architecture is hierarchical and modular, centered around a Root Agent that coordinates between standard business pipelines and recursive RLM workflows.

### Core Components

1. **Root Agent (`root_agent`)**: Acts as the primary coordinator, providing access to both standard pipelines and the RLM recursive workflow.
2. **Vendor Resolution Pipeline**: A `SequentialAgent` that orchestrates:
    - **Parallel ERP Analysis**: Uses a `ParallelAgent` to concurrently analyze data from multiple sources (e.g., Alpha, Beta, Gamma hospital chains).
    - **Vendor Matching**: Resolves vendor instances to master data.
    - **View Generation**: Creates unified analytical views in Databricks.
3. **RLM Workflow**: A specialized `SequentialAgent` implementing the RLM paradigm:
    - **Context Setup**: Prepares the data environment.
    - **RLM Iteration Loop**: A `LoopAgent` that repeatedly:
        - Generates Python code (utilizing `llm_query` for recursion).
        - Executes code in a persistent REPL.
        - Checks for a 'FINAL' answer.
    - **Result Formatter**: Finalizes the output for the user.

### Internal Dependencies & Infrastructure

- **ADK Integration**: Heavily relies on `google.adk` for agent orchestration and tool management.
- **State Management**: Uses `rlm_state.py` and `rlm_repl.py` to maintain session state and REPL persistence across iterations.
- **LLM Bridge**: `llm_bridge.py` provides `llm_query` and `llm_query_batched` functions to the REPL environment, enabling recursive LLM calls from within executed code.

## Mermaid Flowchart

```mermaid
graph TD
    subgraph rlm_adk [RLM-ADK Package]
        Root[root_agent: LlmAgent]
        
        subgraph VRP [vendor_resolution_pipeline: SequentialAgent]
            PEA[parallel_erp_analysis: ParallelAgent]
            VM[vendor_matcher_agent]
            VG[view_generator_agent]
            
            PEA --> VM --> VG
        end
        
        subgraph RWF [rlm_workflow: SequentialAgent]
            CS[context_setup_agent]
            
            subgraph RIL [rlm_iteration_loop: LoopAgent]
                CG[rlm_code_generator]
                CE[rlm_code_executor]
                CC[rlm_completion_checker]
                
                CG --> CE --> CC
                CC -- "Not Final" --> CG
            end
            
            RF[result_formatter_agent]
            
            CS --> RIL --> RF
        end
        
        Root --> VRP
        Root --> RWF
        
        subgraph Tools [Tools & Infrastructure]
            DB[databricks_repl]
            UC[unity_catalog]
            VR[vendor_resolution]
            RT[rlm_tools]
            REPL[rlm_repl / rlm_state]
        end
        
        CE --> RT
        RT --> REPL
        REPL --> DB
        Root --> UC
        Root --> VR
    end
```

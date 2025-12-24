"""MCP Server for AlphaFold3

Provides both synchronous and asynchronous (submit) APIs for AlphaFold3 structure prediction and variant analysis.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("alphafold3")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def prepare_variants(
    variants_fasta: str,
    wt_data_json: str,
    output_dir: str,
    ligand_smiles: Optional[str] = None,
    ligand_id: str = "B",
    model_seeds: Optional[str] = None
) -> dict:
    """
    Prepare variant configurations from a FASTA file and wild-type data.

    Use this to quickly prepare input configurations for AlphaFold3 variant predictions.
    This operation completes in seconds and is suitable for immediate results.

    Args:
        variants_fasta: Path to FASTA file containing variant sequences
        wt_data_json: Path to wild-type AF3 data JSON file (*_data.json)
        output_dir: Directory to save variant configurations
        ligand_smiles: Optional SMILES string for a ligand to include
        ligand_id: Chain ID for the ligand (default: 'B')
        model_seeds: Comma-separated model seeds (e.g., "1,2,3")

    Returns:
        Dictionary with created count, skipped count, and variant directories

    Example:
        Use prepare_variants with:
        - variants_fasta: "path/to/variants.fasta"
        - wt_data_json: "path/to/wt_data.json"
        - output_dir: "variants_output"
    """
    from prepare_variants import prepare_variant_configs

    try:
        # Parse model seeds if provided
        parsed_model_seeds = None
        if model_seeds:
            parsed_model_seeds = [int(s.strip()) for s in model_seeds.split(",")]

        result = prepare_variant_configs(
            variants_fasta=variants_fasta,
            wt_data_json=wt_data_json,
            output_dir=output_dir,
            ligand_smiles=ligand_smiles,
            ligand_id=ligand_id,
            model_seeds=parsed_model_seeds,
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"prepare_variants failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def create_simple_protein_config(
    sequence: str,
    name: str,
    output_dir: str = "output"
) -> dict:
    """
    Create AlphaFold3 configuration for simple protein structure prediction.

    Use this to quickly create input configurations for single protein sequences.
    This operation completes immediately.

    Args:
        sequence: Single-letter amino acid sequence
        name: Name for the prediction job
        output_dir: Directory where config will be saved (default: "output")

    Returns:
        Dictionary with config path and details

    Example:
        Use create_simple_protein_config with:
        - sequence: "MDPSSPNYDKWEMERTDITMKHKLGGGQY..."
        - name: "my_protein"
        - output_dir: "protein_output"
    """
    try:
        # Validate sequence
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        if not all(aa.upper() in valid_aa for aa in sequence):
            return {"status": "error", "error": "Invalid amino acid sequence"}

        # Create config
        config = {
            "name": name,
            "sequences": [
                {
                    "protein": {
                        "id": ["A"],
                        "sequence": sequence
                    }
                }
            ],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1
        }

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Write config
        config_path = output_path / "input.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return {
            "status": "success",
            "config_path": str(config_path),
            "sequence_length": len(sequence),
            "output_dir": str(output_path)
        }
    except Exception as e:
        logger.error(f"create_simple_protein_config failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def validate_fasta_sequences(fasta_path: str) -> dict:
    """
    Validate protein sequences in a FASTA file.

    Use this to quickly check if FASTA sequences are valid for AlphaFold3 prediction.

    Args:
        fasta_path: Path to FASTA file to validate

    Returns:
        Dictionary with validation results and sequence details

    Example:
        Use validate_fasta_sequences with fasta_path "sequences.fasta"
    """
    try:
        from prepare_variants import parse_fasta

        variants = parse_fasta(fasta_path)
        if not variants:
            return {"status": "error", "error": "No sequences found in FASTA file"}

        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        valid_sequences = []
        invalid_sequences = []

        for variant in variants:
            is_valid = all(aa.upper() in valid_aa for aa in variant.sequence)
            seq_info = {
                "name": variant.name,
                "length": len(variant.sequence),
                "sequence_preview": variant.sequence[:50] + ("..." if len(variant.sequence) > 50 else "")
            }

            if is_valid:
                valid_sequences.append(seq_info)
            else:
                invalid_sequences.append(seq_info)

        return {
            "status": "success",
            "total_sequences": len(variants),
            "valid_count": len(valid_sequences),
            "invalid_count": len(invalid_sequences),
            "valid_sequences": valid_sequences,
            "invalid_sequences": invalid_sequences
        }
    except FileNotFoundError:
        return {"status": "error", "error": f"FASTA file not found: {fasta_path}"}
    except Exception as e:
        logger.error(f"validate_fasta_sequences failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_structure_prediction(
    data_path: str,
    device: int = 0,
    model_dir: Optional[str] = None,
    db_dir: Optional[str] = None,
    prediction_mode: str = "default",
    run_template_search: bool = True,
    flash_attention: str = "triton",
    job_name: Optional[str] = None
) -> dict:
    """
    Submit AlphaFold3 structure prediction for background processing.

    This operation can take 30+ minutes for structure prediction. Returns a job_id
    for tracking progress. Use get_job_status() to monitor and get_job_result()
    to retrieve results when completed.

    Args:
        data_path: Directory containing input.json or subdirectories with input.json files
        device: GPU device number (default: 0)
        model_dir: Path to model parameters (default: use AF3 installation)
        db_dir: Path to databases (default: use AF3 installation)
        prediction_mode: "default" (full MSA), "a3m" (precomputed MSA), or "msa" (MSA-only)
        run_template_search: Whether to search for templates (for a3m mode)
        flash_attention: Flash attention implementation ("triton", "cudnn", "xla")
        job_name: Optional name for tracking the job

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs

    Example:
        Submit structure prediction:
        - data_path: "examples/1iep_a3m_fix"
        - prediction_mode: "default"
        - device: 0
    """
    script_path = str(SCRIPTS_DIR / "alphafold3_runner.py")

    # Build arguments based on mode
    args = {
        "mode": prediction_mode,
        "data_path": data_path,
        "device": device,
        "flash_attention": flash_attention
    }

    if model_dir:
        args["model_dir"] = model_dir
    if db_dir:
        args["db_dir"] = db_dir
    if prediction_mode == "a3m":
        args["run_template_search"] = str(run_template_search).lower()

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"structure_prediction_{Path(data_path).name}"
    )

@mcp.tool()
def submit_batch_variants(
    input_dir: str,
    device: int = 0,
    model_dir: Optional[str] = None,
    db_dir: Optional[str] = None,
    run_template_search: bool = False,
    skip_existing: bool = True,
    flash_attention: str = "triton",
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch variant predictions for background processing.

    This operation processes multiple variants efficiently by loading the model once.
    Can take hours for many variants. Returns a job_id for tracking progress.

    Args:
        input_dir: Directory containing variant subdirectories with input.json files
        device: GPU device number (default: 0)
        model_dir: Path to model parameters (default: use AF3 installation)
        db_dir: Path to databases (default: use AF3 installation)
        run_template_search: Whether to search for templates
        skip_existing: Skip variants that already have outputs
        flash_attention: Flash attention implementation ("triton", "cudnn", "xla")
        job_name: Optional name for tracking the job

    Returns:
        Dictionary with job_id for tracking the batch job

    Example:
        Submit batch variant processing:
        - input_dir: "variants_prepared"
        - skip_existing: true
        - run_template_search: false
    """
    script_path = str(SCRIPTS_DIR / "alphafold3_runner.py")

    args = {
        "mode": "batch",
        "data_path": input_dir,
        "device": device,
        "skip_existing": str(skip_existing).lower(),
        "run_template_search": str(run_template_search).lower(),
        "flash_attention": flash_attention
    }

    if model_dir:
        args["model_dir"] = model_dir
    if db_dir:
        args["db_dir"] = db_dir

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_variants_{Path(input_dir).name}"
    )

@mcp.tool()
def submit_prepare_and_predict_variants(
    variants_fasta: str,
    wt_data_json: str,
    output_dir: str,
    device: int = 0,
    ligand_smiles: Optional[str] = None,
    skip_existing: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit end-to-end variant analysis: prepare configs and run predictions.

    This operation combines variant preparation and batch prediction into a single
    long-running job. Can take hours depending on the number of variants.

    Args:
        variants_fasta: Path to FASTA file containing variant sequences
        wt_data_json: Path to wild-type AF3 data JSON file
        output_dir: Directory for variant configs and predictions
        device: GPU device number (default: 0)
        ligand_smiles: Optional SMILES string for ligand
        skip_existing: Skip already completed predictions
        job_name: Optional name for tracking the job

    Returns:
        Dictionary with job_id for tracking the complete workflow

    Example:
        Submit complete variant workflow:
        - variants_fasta: "variants.fasta"
        - wt_data_json: "wt_protein_data.json"
        - output_dir: "variant_analysis"
    """
    script_path = str(SCRIPTS_DIR / "prepare_variants.py")

    args = {
        "variants_fasta": variants_fasta,
        "wt_data_json": wt_data_json,
        "output_dir": output_dir,
        "run_prediction": "true",  # Custom flag to run prediction after prep
        "device": device,
        "skip_existing": str(skip_existing).lower()
    }

    if ligand_smiles:
        args["ligand_smiles"] = ligand_smiles

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"variants_workflow_{Path(output_dir).name}"
    )

# ==============================================================================
# Information and Utility Tools
# ==============================================================================

@mcp.tool()
def get_server_info() -> dict:
    """
    Get information about the AlphaFold3 MCP server.

    Returns:
        Dictionary with server capabilities and tool information
    """
    return {
        "server_name": "alphafold3",
        "version": "1.0.0",
        "description": "MCP server for AlphaFold3 structure prediction and variant analysis",
        "sync_tools": [
            "prepare_variants",
            "create_simple_protein_config",
            "validate_fasta_sequences"
        ],
        "submit_tools": [
            "submit_structure_prediction",
            "submit_batch_variants",
            "submit_prepare_and_predict_variants"
        ],
        "job_management": [
            "get_job_status",
            "get_job_result",
            "get_job_log",
            "cancel_job",
            "list_jobs"
        ],
        "scripts_directory": str(SCRIPTS_DIR),
        "jobs_directory": str(job_manager.jobs_dir)
    }

@mcp.tool()
def get_example_workflows() -> dict:
    """
    Get example workflows for common AlphaFold3 use cases.

    Returns:
        Dictionary with example workflow descriptions
    """
    return {
        "workflows": {
            "simple_protein": {
                "description": "Predict structure for a single protein sequence",
                "steps": [
                    "1. Use create_simple_protein_config() with your protein sequence",
                    "2. Use submit_structure_prediction() with the created config directory",
                    "3. Monitor with get_job_status() and get results with get_job_result()"
                ],
                "example": "create_simple_protein_config(sequence='MDPSS...', name='kinase')"
            },
            "protein_variants": {
                "description": "Analyze multiple protein variants efficiently",
                "steps": [
                    "1. Prepare a FASTA file with all variant sequences",
                    "2. Get wild-type data JSON from previous AF3 run",
                    "3. Use submit_prepare_and_predict_variants() for end-to-end analysis",
                    "4. Monitor progress and get batch results"
                ],
                "example": "submit_prepare_and_predict_variants(variants_fasta='variants.fasta', wt_data_json='wt_data.json')"
            },
            "batch_processing": {
                "description": "Process pre-configured input directories",
                "steps": [
                    "1. Prepare input.json files in subdirectories",
                    "2. Use submit_batch_variants() to process all at once",
                    "3. Monitor batch progress and get individual results"
                ],
                "example": "submit_batch_variants(input_dir='prepared_variants/', skip_existing=True)"
            }
        },
        "data_requirements": {
            "simple_prediction": "Just a protein sequence",
            "variant_analysis": "Variant FASTA + wild-type AF3 data JSON",
            "batch_processing": "Directory with input.json files"
        }
    }

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()
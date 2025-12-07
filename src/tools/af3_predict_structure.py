"""
AlphaFold3 structure prediction tools.

This MCP Server provides tools for running AlphaFold3 structure predictions:
1. af3_predict_structure: Run AlphaFold3 structure prediction with various modes
2. af3_predict_batch: Run batch AlphaFold3 predictions on multiple inputs
3. af3_predict_structure_from_seq: Run full pipeline from sequences (protein, DNA, RNA, ligand)

The tools support different prediction modes:
- default: Full MSA search + template search + inference
- a3m: Pre-computed A3M MSA + optional template search + inference
- msa: Pre-computed MSA/templates (inference only, fastest)
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated, Literal, Optional

from fastmcp import FastMCP
from loguru import logger

# MCP server instance
af3_predict_mcp = FastMCP(name="af3_predict_structure")


def _get_af3_path() -> Path:
    """Get the AlphaFold3 repository path."""
    script_dir = Path(__file__).parent.parent.parent.absolute()
    af3_path = script_dir / "repo" / "alphafold3"

    if not af3_path.exists():
        logger.error(f"AlphaFold3 repository not found at {af3_path}")
        raise FileNotFoundError(
            f"AlphaFold3 repository not found at {af3_path}. "
            "Please clone the AlphaFold3 repository to repo/alphafold3."
        )
    return af3_path


def _get_default_paths(af3_path: Path) -> dict:
    """Get default model and database paths."""
    return {
        "model_dir": str(af3_path / "model"),
        "db_dir": str(af3_path / "alphafold3_db"),
    }


def _resolve_path(path: Optional[str]) -> Optional[str]:
    """Resolve a path to absolute."""
    if path is None:
        return None
    return str(Path(path).resolve())


def _log_stream(stream, logs: list[str], prefix: str = ""):
    """Collect output from a stream and print in real-time."""
    for line in iter(stream.readline, ""):
        line = line.rstrip()
        if line:
            logs.append(line)
            # Print in real-time with prefix
            logger.info(f"{prefix}{line}")


def _run_command(
    cmd: list[str],
    device: int = 0,
    cwd: Optional[str] = None,
) -> dict:
    """Run a command with proper environment setup."""
    # Set up environment
    run_env = os.environ.copy()
    run_env["CUDA_VISIBLE_DEVICES"] = str(device)
    run_env["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
    run_env["PYTHONUNBUFFERED"] = "1"

    # Get output directory from command args
    output_dir = ""
    for i, arg in enumerate(cmd):
        if arg.startswith("--output_dir="):
            output_dir = arg.split("=", 1)[1]
            break
        elif arg == "--output_dir" and i + 1 < len(cmd):
            output_dir = cmd[i + 1]
            break

    # Collect logs
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    logger.debug(f"Executing command: {' '.join(cmd)}")

    # Run the command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=run_env,
        cwd=cwd,
        bufsize=1,
    )

    # Read stdout and stderr with real-time logging
    stdout_thread = threading.Thread(
        target=_log_stream,
        args=(process.stdout, stdout_lines, "[AF3 stdout] "),
    )
    stderr_thread = threading.Thread(
        target=_log_stream,
        args=(process.stderr, stderr_lines, "[AF3 stderr] "),
    )

    stdout_thread.start()
    stderr_thread.start()

    process.wait()
    stdout_thread.join()
    stderr_thread.join()

    if process.stdout:
        process.stdout.close()
    if process.stderr:
        process.stderr.close()

    logger.debug(f"Command completed with return code: {process.returncode}")

    return {
        "success": process.returncode == 0,
        "output_dir": output_dir,
        "return_code": process.returncode,
        "stdout": "\n".join(stdout_lines),
        "stderr": "\n".join(stderr_lines),
    }


@af3_predict_mcp.tool
def af3_predict_structure(
    data_path: Annotated[str, "Path to directory containing input.json or path to input.json file"],
    mode: Annotated[
        Literal["default", "a3m", "msa"],
        "Prediction mode: 'default' (full pipeline), 'a3m' (with pre-computed A3M), 'msa' (inference only)"
    ] = "default",
    device: Annotated[int, "GPU device number"] = 0,
    model_dir: Annotated[Optional[str], "Path to model parameters. If None, uses default"] = None,
    db_dir: Annotated[Optional[str], "Path to databases. If None, uses default"] = None,
    run_template_search: Annotated[bool, "Whether to search for templates (a3m mode only)"] = True,
    max_template_date: Annotated[str, "Maximum template release date in YYYY-MM-DD format"] = "2021-09-30",
    flash_attention_implementation: Annotated[
        Literal["triton", "cudnn", "xla"],
        "Flash attention implementation. Use 'xla' for non-A100 GPUs"
    ] = "triton",
    buckets: Annotated[Optional[str], "Comma-separated token bucket sizes for compilation caching"] = None,
) -> dict:
    """
    Run AlphaFold3 structure prediction.

    This tool supports three prediction modes:

    1. **default**: Full pipeline with MSA search + template search + inference
       - Input: input.json with protein sequence
       - Slowest but most accurate for novel proteins

    2. **a3m**: Use pre-computed A3M MSA files
       - Input: input.json with msa_path references to A3M files
       - Optionally runs template search
       - Good when you have MSAs from external sources (MMseqs2, HHblits)

    3. **msa**: Inference only with embedded MSA/templates
       - Input: input.json with inline unpairedMsa, pairedMsa, and templates
       - Fastest mode for variant analysis
       - Skip all search, run inference directly

    Input: Path to directory with input.json or path to input.json file
    Output: Dictionary with prediction status, output paths, and logs
    """
    logger.info(f"af3_predict_structure called with data_path={data_path}, mode={mode}, device={device}")

    try:
        af3_path = _get_af3_path()
        defaults = _get_default_paths(af3_path)

        # Resolve paths
        data_path = Path(data_path).resolve()
        model_dir = _resolve_path(model_dir) or defaults["model_dir"]
        db_dir = _resolve_path(db_dir) or defaults["db_dir"]

        # Find input.json
        if data_path.is_file() and data_path.name == "input.json":
            input_json = data_path
            output_dir = data_path.parent
        else:
            input_json = data_path / "input.json"
            output_dir = data_path

        if not input_json.exists():
            logger.error(f"input.json not found at {input_json}")
            raise FileNotFoundError(f"input.json not found at {input_json}")

        logger.info(f"Input JSON: {input_json}, Output dir: {output_dir}")

        # Build command based on mode
        if mode == "default":
            cmd = [
                sys.executable,
                str(af3_path / "run_alphafold.py"),
                f"--json_path={input_json}",
                f"--model_dir={model_dir}",
                f"--db_dir={db_dir}",
                f"--output_dir={output_dir}",
                "--run_data_pipeline=true",
                "--run_inference=true",
                f"--flash_attention_implementation={flash_attention_implementation}",
            ]
        elif mode == "a3m":
            cmd = [
                sys.executable,
                str(af3_path / "run_alphafold_with_a3m.py"),
                f"--input_json={input_json}",
                f"--model_dir={model_dir}",
                f"--db_dir={db_dir}",
                f"--output_dir={output_dir}",
                f"--run_template_search={str(run_template_search).lower()}",
                f"--max_template_date={max_template_date}",
                "--run_inference=true",
                f"--flash_attention_implementation={flash_attention_implementation}",
            ]
        elif mode == "msa":
            cmd = [
                sys.executable,
                str(af3_path / "run_alphafold.py"),
                f"--json_path={input_json}",
                f"--model_dir={model_dir}",
                f"--db_dir={db_dir}",
                f"--output_dir={output_dir}",
                "--run_data_pipeline=false",
                "--run_inference=true",
                f"--flash_attention_implementation={flash_attention_implementation}",
            ]
        else:
            logger.error(f"Unknown mode: {mode}")
            raise ValueError(f"Unknown mode: {mode}")

        if buckets:
            cmd.append(f"--buckets={buckets}")

        logger.info(f"Running prediction on GPU {device} in '{mode}' mode")

        # Run prediction
        result = _run_command(cmd, device=device, cwd=str(af3_path))

        # Find output files
        output_files = []
        if result["success"]:
            output_path = Path(result["output_dir"])
            if output_path.exists():
                output_files = [
                    str(f.relative_to(output_path))
                    for f in output_path.glob("*")
                    if f.is_file()
                ]
            logger.info(f"Prediction completed successfully. Output files: {len(output_files)}")
        else:
            logger.error(f"Prediction failed with return code {result['return_code']}")
            logger.error(f"stderr: {result['stderr'][-1000:] if len(result['stderr']) > 1000 else result['stderr']}")

        return {
            "status": "success" if result["success"] else "error",
            "mode": mode,
            "input_json": str(input_json),
            "output_dir": result["output_dir"],
            "output_files": output_files,
            "return_code": result["return_code"],
            "device": device,
            "flash_attention": flash_attention_implementation,
            "stdout_preview": result["stdout"][-2000:] if len(result["stdout"]) > 2000 else result["stdout"],
            "stderr_preview": result["stderr"][-2000:] if len(result["stderr"]) > 2000 else result["stderr"],
        }

    except Exception as e:
        logger.exception(f"Exception during prediction: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "data_path": str(data_path),
            "mode": mode,
        }


def _af3_predict_batch_impl(
    input_dir: str,
    device: int = 0,
    model_dir: Optional[str] = None,
    db_dir: Optional[str] = None,
    run_template_search: bool = True,
    max_template_date: str = "2021-09-30",
    skip_existing: bool = True,
    output_marker: str = "model_output.cif",
    flash_attention_implementation: str = "triton",
    buckets: Optional[str] = None,
) -> dict:
    """Internal implementation of af3_predict_batch."""
    logger.info(f"af3_predict_batch called with input_dir={input_dir}, device={device}")

    try:
        af3_path = _get_af3_path()
        defaults = _get_default_paths(af3_path)

        # Resolve paths
        input_dir = Path(input_dir).resolve()
        model_dir = _resolve_path(model_dir) or defaults["model_dir"]
        db_dir = _resolve_path(db_dir) or defaults["db_dir"]

        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Count inputs
        input_jsons = list(input_dir.glob("*/input.json"))
        total_inputs = len(input_jsons)

        if total_inputs == 0:
            logger.error(f"No input.json files found in subdirectories of {input_dir}")
            raise ValueError(f"No input.json files found in subdirectories of {input_dir}")

        logger.info(f"Found {total_inputs} input files to process")

        # Build batch command
        cmd = [
            sys.executable,
            str(af3_path / "run_alphafold_with_a3m_batch.py"),
            f"--batch_input_dir={input_dir}",
            f"--model_dir={model_dir}",
            f"--db_dir={db_dir}",
            f"--run_template_search={str(run_template_search).lower()}",
            f"--max_template_date={max_template_date}",
            f"--skip_existing={str(skip_existing).lower()}",
            f"--output_marker={output_marker}",
            "--input_json_name=input.json",
            "--run_inference=true",
            f"--flash_attention_implementation={flash_attention_implementation}",
        ]

        if buckets:
            cmd.append(f"--buckets={buckets}")

        logger.info(f"Running batch prediction on GPU {device}")

        # Run batch prediction
        result = _run_command(cmd, device=device, cwd=str(af3_path))

        # Count completed predictions
        completed = []
        for input_json in input_jsons:
            variant_dir = input_json.parent
            output_file = variant_dir / output_marker
            if output_file.exists():
                completed.append(variant_dir.name)

        if result["success"]:
            logger.info(f"Batch prediction completed successfully. Completed: {len(completed)}/{total_inputs}")
        else:
            logger.error(f"Batch prediction failed with return code {result['return_code']}")
            logger.error(f"stderr: {result['stderr'][-1000:] if len(result['stderr']) > 1000 else result['stderr']}")

        return {
            "status": "success" if result["success"] else "error",
            "input_dir": str(input_dir),
            "total_inputs": total_inputs,
            "completed_count": len(completed),
            "completed_variants": completed[:20] if len(completed) > 20 else completed,
            "return_code": result["return_code"],
            "device": device,
            "skip_existing": skip_existing,
            "flash_attention": flash_attention_implementation,
            "stdout_preview": result["stdout"][-2000:] if len(result["stdout"]) > 2000 else result["stdout"],
            "stderr_preview": result["stderr"][-2000:] if len(result["stderr"]) > 2000 else result["stderr"],
        }

    except Exception as e:
        logger.exception(f"Exception during batch prediction: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "input_dir": str(input_dir),
        }


@af3_predict_mcp.tool
def af3_predict_batch(
    input_dir: Annotated[str, "Directory containing variant subdirectories with input.json files"],
    device: Annotated[int, "GPU device number"] = 0,
    model_dir: Annotated[Optional[str], "Path to model parameters. If None, uses default"] = None,
    db_dir: Annotated[Optional[str], "Path to databases. If None, uses default"] = None,
    run_template_search: Annotated[bool, "Whether to search templates (only for msa_path inputs)"] = True,
    max_template_date: Annotated[str, "Maximum template release date in YYYY-MM-DD format"] = "2021-09-30",
    skip_existing: Annotated[bool, "Skip predictions if output files already exist"] = True,
    output_marker: Annotated[str, "File to check for determining if prediction is complete"] = "model_output.cif",
    flash_attention_implementation: Annotated[
        Literal["triton", "cudnn", "xla"],
        "Flash attention implementation. Use 'xla' for non-A100 GPUs"
    ] = "triton",
    buckets: Annotated[Optional[str], "Comma-separated token bucket sizes for compilation caching"] = None,
) -> dict:
    """
    Run AlphaFold3 batch predictions on multiple inputs.

    This tool processes multiple variant directories efficiently:
    - Model is loaded once and reused for all predictions
    - Auto-detects input format (inline MSA/templates vs msa_path references)
    - Can skip already completed predictions

    Expected directory structure:
        input_dir/
        ├── variant1/
        │   └── input.json
        ├── variant2/
        │   └── input.json
        └── variant3/
            └── input.json

    Input: Directory path containing variant subdirectories
    Output: Dictionary with batch status, processed counts, and logs
    """
    return _af3_predict_batch_impl(
        input_dir=input_dir,
        device=device,
        model_dir=model_dir,
        db_dir=db_dir,
        run_template_search=run_template_search,
        max_template_date=max_template_date,
        skip_existing=skip_existing,
        output_marker=output_marker,
        flash_attention_implementation=flash_attention_implementation,
        buckets=buckets,
    )


def _create_af3_input_json(
    name: str,
    protein_seqs: Optional[list[str]] = None,
    dna_seqs: Optional[list[str]] = None,
    rna_seqs: Optional[list[str]] = None,
    ligand_smiles: Optional[list[str]] = None,
    ligand_ccd_ids: Optional[list[str]] = None,
    model_seeds: list[int] = None,
) -> dict:
    """
    Create AlphaFold3 input JSON from sequences.

    Args:
        name: Name of the prediction job
        protein_seqs: List of protein sequences (amino acid single-letter codes)
        dna_seqs: List of DNA sequences (ACGT)
        rna_seqs: List of RNA sequences (ACGU)
        ligand_smiles: List of ligand SMILES strings
        ligand_ccd_ids: List of ligand CCD IDs
        model_seeds: List of random seeds for predictions

    Returns:
        AlphaFold3 input JSON dictionary
    """
    if model_seeds is None:
        model_seeds = [1]

    sequences = []
    chain_id_counter = 0

    def get_chain_id(index: int) -> str:
        """Generate chain ID from index (A, B, C, ...)."""
        return chr(ord('A') + index)

    # Add protein chains
    if protein_seqs:
        for seq in protein_seqs:
            chain_id = get_chain_id(chain_id_counter)
            sequences.append({
                "protein": {
                    "id": [chain_id],
                    "sequence": seq.upper().replace(" ", "").replace("\n", ""),
                }
            })
            chain_id_counter += 1

    # Add DNA chains
    if dna_seqs:
        for seq in dna_seqs:
            chain_id = get_chain_id(chain_id_counter)
            sequences.append({
                "dna": {
                    "id": [chain_id],
                    "sequence": seq.upper().replace(" ", "").replace("\n", ""),
                }
            })
            chain_id_counter += 1

    # Add RNA chains
    if rna_seqs:
        for seq in rna_seqs:
            chain_id = get_chain_id(chain_id_counter)
            sequences.append({
                "rna": {
                    "id": [chain_id],
                    "sequence": seq.upper().replace(" ", "").replace("\n", ""),
                }
            })
            chain_id_counter += 1

    # Add ligands (SMILES)
    if ligand_smiles:
        for smiles in ligand_smiles:
            chain_id = get_chain_id(chain_id_counter)
            sequences.append({
                "ligand": {
                    "id": [chain_id],
                    "smiles": smiles,
                }
            })
            chain_id_counter += 1

    # Add ligands (CCD IDs)
    if ligand_ccd_ids:
        for ccd_id in ligand_ccd_ids:
            chain_id = get_chain_id(chain_id_counter)
            sequences.append({
                "ligand": {
                    "id": [chain_id],
                    "ccdCodes": [ccd_id],
                }
            })
            chain_id_counter += 1

    return {
        "name": name,
        "sequences": sequences,
        "modelSeeds": model_seeds,
        "dialect": "alphafold3",
        "version": 1,
    }


@af3_predict_mcp.tool
def af3_predict_structure_from_seq(
    name: Annotated[str, "Name of the prediction job"],
    output_dir: Annotated[str, "Output directory for prediction results"],
    protein_seqs: Annotated[Optional[str], "Protein sequences separated by '|' (e.g., 'MKLL...|MDEF...'). Set to None if not needed"] = None,
    dna_seqs: Annotated[Optional[str], "DNA sequences separated by '|' (e.g., 'ACGT...|TGCA...'). Set to None if not needed"] = None,
    rna_seqs: Annotated[Optional[str], "RNA sequences separated by '|' (e.g., 'ACGU...|UGCA...'). Set to None if not needed"] = None,
    ligand_smiles: Annotated[Optional[str], "Ligand SMILES strings separated by '|'. Set to None if not needed"] = None,
    ligand_ccd_ids: Annotated[Optional[str], "Ligand CCD IDs separated by '|' (e.g., 'ATP|MG'). Set to None if not needed"] = None,
    model_seeds: Annotated[str, "Comma-separated model seeds (e.g., '1,2,3')"] = "1",
    device: Annotated[int, "GPU device number"] = 0,
    model_dir: Annotated[Optional[str], "Path to model parameters. If None, uses default"] = None,
    db_dir: Annotated[Optional[str], "Path to databases. If None, uses default"] = None,
    flash_attention_implementation: Annotated[
        Literal["triton", "cudnn", "xla"],
        "Flash attention implementation. Use 'xla' for non-A100 GPUs"
    ] = "triton",
    buckets: Annotated[Optional[str], "Comma-separated token bucket sizes for compilation caching"] = None,
) -> dict:
    """
    Run AlphaFold3 structure prediction from sequences.

    This tool runs the full AlphaFold3 pipeline including:
    1. Creating input.json configuration from provided sequences
    2. Running the data pipeline (MSA search, template search)
    3. Running structure inference

    Supports multiple chain types:
    - **Protein**: Standard amino acid sequences (single-letter codes)
    - **DNA**: DNA sequences (A, C, G, T)
    - **RNA**: RNA sequences (A, C, G, U)
    - **Ligand (SMILES)**: Small molecules defined by SMILES strings
    - **Ligand (CCD)**: Small molecules defined by CCD IDs (e.g., ATP, MG)

    Each chain type is optional. Set to None to exclude from prediction.
    Multiple sequences of the same type are separated by '|'.

    Example:
        - Protein-ligand complex: protein_seqs="MKLL...", ligand_smiles="CCO"
        - Protein-DNA complex: protein_seqs="MKLL...", dna_seqs="ACGTACGT"
        - Multi-chain protein: protein_seqs="MKLL...|MDEF..."

    Input: Sequences for each chain type
    Output: Dictionary with prediction status, output paths, and logs
    """
    logger.info(f"af3_predict_structure_from_seq called with name={name}")

    try:
        # Validate that at least one sequence type is provided
        if all(x is None for x in [protein_seqs, dna_seqs, rna_seqs, ligand_smiles, ligand_ccd_ids]):
            logger.error("At least one sequence type must be provided")
            raise ValueError("At least one sequence type must be provided (protein_seqs, dna_seqs, rna_seqs, ligand_smiles, or ligand_ccd_ids)")

        af3_path = _get_af3_path()
        defaults = _get_default_paths(af3_path)

        # Resolve paths
        output_dir = Path(output_dir).resolve()
        model_dir = _resolve_path(model_dir) or defaults["model_dir"]
        db_dir = _resolve_path(db_dir) or defaults["db_dir"]

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Parse sequences (split by '|')
        protein_list = protein_seqs.split("|") if protein_seqs else None
        dna_list = dna_seqs.split("|") if dna_seqs else None
        rna_list = rna_seqs.split("|") if rna_seqs else None
        smiles_list = ligand_smiles.split("|") if ligand_smiles else None
        ccd_list = ligand_ccd_ids.split("|") if ligand_ccd_ids else None

        # Parse model seeds
        seeds = [int(s.strip()) for s in model_seeds.split(",")]

        # Log input summary
        logger.info(f"Input summary: "
                    f"proteins={len(protein_list) if protein_list else 0}, "
                    f"dna={len(dna_list) if dna_list else 0}, "
                    f"rna={len(rna_list) if rna_list else 0}, "
                    f"ligands(SMILES)={len(smiles_list) if smiles_list else 0}, "
                    f"ligands(CCD)={len(ccd_list) if ccd_list else 0}")

        # Create input JSON
        input_config = _create_af3_input_json(
            name=name,
            protein_seqs=protein_list,
            dna_seqs=dna_list,
            rna_seqs=rna_list,
            ligand_smiles=smiles_list,
            ligand_ccd_ids=ccd_list,
            model_seeds=seeds,
        )

        # Write input.json
        input_json_path = output_dir / "input.json"
        with open(input_json_path, "w") as f:
            json.dump(input_config, f, indent=2)
        logger.info(f"Created input.json at {input_json_path}")

        # Build command for full pipeline
        cmd = [
            sys.executable,
            str(af3_path / "run_alphafold.py"),
            f"--json_path={input_json_path}",
            f"--model_dir={model_dir}",
            f"--db_dir={db_dir}",
            f"--output_dir={output_dir}",
            "--run_data_pipeline=true",
            "--run_inference=true",
            f"--flash_attention_implementation={flash_attention_implementation}",
        ]

        if buckets:
            cmd.append(f"--buckets={buckets}")

        logger.info(f"Running full pipeline prediction on GPU {device}")

        # Run prediction
        result = _run_command(cmd, device=device, cwd=str(af3_path))

        # Find output files
        output_files = []
        if result["success"]:
            if output_dir.exists():
                output_files = [
                    str(f.relative_to(output_dir))
                    for f in output_dir.glob("*")
                    if f.is_file()
                ]
            logger.info(f"Prediction completed successfully. Output files: {len(output_files)}")
        else:
            logger.error(f"Prediction failed with return code {result['return_code']}")
            logger.error(f"stderr: {result['stderr'][-1000:] if len(result['stderr']) > 1000 else result['stderr']}")

        # Build chain summary
        chain_summary = []
        chain_idx = 0
        if protein_list:
            for seq in protein_list:
                chain_summary.append({"chain": chr(ord('A') + chain_idx), "type": "protein", "length": len(seq)})
                chain_idx += 1
        if dna_list:
            for seq in dna_list:
                chain_summary.append({"chain": chr(ord('A') + chain_idx), "type": "dna", "length": len(seq)})
                chain_idx += 1
        if rna_list:
            for seq in rna_list:
                chain_summary.append({"chain": chr(ord('A') + chain_idx), "type": "rna", "length": len(seq)})
                chain_idx += 1
        if smiles_list:
            for smiles in smiles_list:
                chain_summary.append({"chain": chr(ord('A') + chain_idx), "type": "ligand_smiles", "smiles": smiles})
                chain_idx += 1
        if ccd_list:
            for ccd in ccd_list:
                chain_summary.append({"chain": chr(ord('A') + chain_idx), "type": "ligand_ccd", "ccd_id": ccd})
                chain_idx += 1

        return {
            "status": "success" if result["success"] else "error",
            "name": name,
            "input_json": str(input_json_path),
            "output_dir": str(output_dir),
            "output_files": output_files,
            "chains": chain_summary,
            "model_seeds": seeds,
            "return_code": result["return_code"],
            "device": device,
            "flash_attention": flash_attention_implementation,
            "stdout_preview": result["stdout"][-2000:] if len(result["stdout"]) > 2000 else result["stdout"],
            "stderr_preview": result["stderr"][-2000:] if len(result["stderr"]) > 2000 else result["stderr"],
        }

    except Exception as e:
        logger.exception(f"Exception during prediction from sequences: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "name": name,
        }

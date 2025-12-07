#!/usr/bin/env python3
"""Python interface for running AlphaFold3 structure predictions.

This module provides Python functions to call the AlphaFold3 prediction scripts
with different configurations:

1. Default mode: Full MSA search + template search + inference
2. A3M mode: Pre-computed A3M MSA + optional template search + inference
3. MSA-only mode: Pre-computed MSA/templates (skip all search) + inference
4. Batch mode: Process multiple variants efficiently (model loaded once)

Example usage:
    from alphafold3_runner import (
        run_default,
        run_with_a3m,
        run_with_msa,
        run_batch,
    )

    # Default full pipeline
    run_default("/path/to/data", device=0)

    # With pre-computed A3M
    run_with_a3m("/path/to/data", device=0, run_template_search=True)

    # MSA-only mode (fastest when MSA+templates are pre-computed)
    run_with_msa("/path/to/data", device=0)

    # Batch processing for variants
    run_batch("/path/to/variants", device=0, skip_existing=True)
"""

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from loguru import logger


# Configure loguru for AF3 output
def _configure_logger():
    """Configure loguru logger for AF3 output."""
    # Remove default handler and add custom format
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>AF3</cyan> | {message}",
        level="DEBUG",
        colorize=True,
    )


# Initialize logger configuration
_configure_logger()


# Get the AlphaFold3 repository path
def _get_af3_path() -> Path:
    """Get the AlphaFold3 repository path."""
    # Default: relative to this script's location
    script_dir = Path(__file__).parent.absolute()
    af3_path = script_dir.parent / "repo" / "alphafold3"

    if not af3_path.exists():
        raise FileNotFoundError(
            f"AlphaFold3 repository not found at {af3_path}. "
            "Please set the AF3_PATH environment variable."
        )
    return af3_path


def _get_default_paths(af3_path: Path) -> dict:
    """Get default model and database paths."""
    return {
        "model_dir": af3_path / "model",
        "db_dir": af3_path / "alphafold3_db",
    }


@dataclass
class PredictionResult:
    """Result of an AlphaFold3 prediction run."""

    success: bool
    output_dir: str
    return_code: int
    stdout: str
    stderr: str
    logs: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.success


def _log_stream(stream, level: str, logs: list[str], prefix: str = ""):
    """Log output from a stream line by line using loguru."""
    for line in iter(stream.readline, ""):
        line = line.rstrip()
        if line:
            logs.append(line)
            msg = f"{prefix}{line}" if prefix else line

            # Determine log level based on content
            if any(kw in line.lower() for kw in ["error", "failed", "exception", "traceback"]):
                logger.error(msg)
            elif any(kw in line.lower() for kw in ["warning", "warn"]):
                logger.warning(msg)
            elif any(kw in line.lower() for kw in ["processing", "running", "predicting", "featurising"]):
                logger.info(msg)
            elif any(kw in line.lower() for kw in ["done", "complete", "success", "saved", "wrote"]):
                logger.success(msg)
            elif level == "stderr":
                logger.warning(msg)
            else:
                logger.debug(msg)


def _run_command(
    cmd: list[str],
    device: int = 0,
    env: Optional[dict] = None,
    cwd: Optional[Union[str, Path]] = None,
    job_name: str = "",
) -> PredictionResult:
    """Run a command with proper environment setup and real-time logging."""
    # Set up environment
    run_env = os.environ.copy()
    run_env["CUDA_VISIBLE_DEVICES"] = str(device)
    run_env["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
    # Ensure Python output is unbuffered for real-time logging
    run_env["PYTHONUNBUFFERED"] = "1"

    if env:
        run_env.update(env)

    # Get output directory from command args if present
    output_dir = ""
    for i, arg in enumerate(cmd):
        if arg.startswith("--output_dir="):
            output_dir = arg.split("=", 1)[1]
            break
        elif arg == "--output_dir" and i + 1 < len(cmd):
            output_dir = cmd[i + 1]
            break

    # Log the command being run
    prefix = f"[{job_name}] " if job_name else ""
    logger.info(f"{prefix}Starting AlphaFold3 prediction...")
    logger.debug(f"{prefix}Command: {' '.join(cmd[:3])}...")
    logger.debug(f"{prefix}Output directory: {output_dir}")
    logger.debug(f"{prefix}GPU device: {device}")

    # Collect all logs
    all_logs: list[str] = []
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    # Run the command with real-time output streaming
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=run_env,
        cwd=cwd,
        bufsize=1,  # Line buffered
    )

    # Create threads to read stdout and stderr concurrently
    stdout_thread = threading.Thread(
        target=_log_stream,
        args=(process.stdout, "stdout", stdout_lines, prefix),
    )
    stderr_thread = threading.Thread(
        target=_log_stream,
        args=(process.stderr, "stderr", stderr_lines, prefix),
    )

    stdout_thread.start()
    stderr_thread.start()

    # Wait for process to complete
    process.wait()

    # Wait for logging threads to finish
    stdout_thread.join()
    stderr_thread.join()

    # Close streams
    if process.stdout:
        process.stdout.close()
    if process.stderr:
        process.stderr.close()

    # Combine logs
    all_logs = stdout_lines + stderr_lines

    # Log final status
    if process.returncode == 0:
        logger.success(f"{prefix}AlphaFold3 prediction completed successfully!")
    else:
        logger.error(f"{prefix}AlphaFold3 prediction failed with return code {process.returncode}")

    return PredictionResult(
        success=process.returncode == 0,
        output_dir=output_dir,
        return_code=process.returncode,
        stdout="\n".join(stdout_lines),
        stderr="\n".join(stderr_lines),
        logs=all_logs,
    )


def _resolve_path(path: Union[str, Path, None]) -> Optional[Path]:
    """Resolve a path to absolute, handling None values."""
    if path is None:
        return None
    return Path(path).resolve()


def run_default(
    data_path: Union[str, Path],
    device: int = 0,
    model_dir: Optional[Union[str, Path]] = None,
    db_dir: Optional[Union[str, Path]] = None,
    flash_attention_implementation: str = "triton",
    buckets: Optional[list[int]] = None,
    jax_compilation_cache_dir: Optional[Union[str, Path]] = None,
) -> list[PredictionResult]:
    """Run AlphaFold3 with default configuration (full MSA + template search).

    This runs the complete data pipeline including MSA search and template search,
    followed by model inference.

    Args:
        data_path: Directory containing subdirectories with input.json files,
                   or path to a single directory with input.json.
        device: GPU device number (default: 0).
        model_dir: Path to model parameters. If None, uses default.
        db_dir: Path to databases. If None, uses default.
        flash_attention_implementation: 'triton', 'cudnn', or 'xla' (default: 'triton').
                                        Use 'xla' for non-A100 GPUs.
        buckets: Token bucket sizes for compilation caching.
        jax_compilation_cache_dir: Directory for JAX compilation cache.

    Returns:
        List of PredictionResult objects, one per input.json processed.

    Example:
        # Process all inputs in a directory
        results = run_default("/path/to/data", device=0)

        # Process with custom paths
        results = run_default(
            "/path/to/data",
            model_dir="/custom/model",
            db_dir="/custom/db",
        )
    """
    af3_path = _get_af3_path()
    defaults = _get_default_paths(af3_path)

    # Resolve all paths to absolute paths (important since we change cwd to af3_path)
    data_path = Path(data_path).resolve()
    model_dir = _resolve_path(model_dir) or defaults["model_dir"]
    db_dir = _resolve_path(db_dir) or defaults["db_dir"]
    jax_compilation_cache_dir = _resolve_path(jax_compilation_cache_dir)

    # Find all input.json files
    if data_path.is_file() and data_path.name == "input.json":
        input_jsons = [data_path]
    elif (data_path / "input.json").exists():
        input_jsons = [data_path / "input.json"]
    else:
        input_jsons = sorted(data_path.glob("*/input.json"))

    if not input_jsons:
        raise FileNotFoundError(f"No input.json files found in {data_path}")

    logger.info(f"Found {len(input_jsons)} input(s) to process")

    results = []
    for idx, input_json in enumerate(input_jsons, 1):
        output_dir = input_json.parent
        job_name = output_dir.name

        logger.info(f"[{idx}/{len(input_jsons)}] Processing {job_name}...")

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

        if buckets:
            cmd.append(f"--buckets={','.join(map(str, buckets))}")

        if jax_compilation_cache_dir:
            cmd.append(f"--jax_compilation_cache_dir={jax_compilation_cache_dir}")

        result = _run_command(cmd, device=device, cwd=af3_path, job_name=job_name)
        result.output_dir = str(output_dir)
        results.append(result)

    # Summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"Completed: {successful}/{len(results)} predictions successful")

    return results


def run_with_a3m(
    data_path: Union[str, Path],
    device: int = 0,
    model_dir: Optional[Union[str, Path]] = None,
    db_dir: Optional[Union[str, Path]] = None,
    run_template_search: bool = True,
    max_template_date: str = "2021-09-30",
    unpaired_msa_a3m: Optional[Union[str, Path]] = None,
    paired_msa_a3m: Optional[Union[str, Path]] = None,
    unpaired_msa_dir: Optional[Union[str, Path]] = None,
    paired_msa_dir: Optional[Union[str, Path]] = None,
    run_inference: bool = True,
    flash_attention_implementation: str = "triton",
    buckets: Optional[list[int]] = None,
) -> PredictionResult:
    """Run AlphaFold3 with pre-computed A3M MSA files.

    This skips MSA search but can optionally run template search using the
    provided MSA. Use this when you have pre-computed MSAs from external
    sources (e.g., MMseqs2, HHblits).

    A3M files can be provided via:
    1. msa_path field in input.json (relative or absolute path)
    2. Command-line arguments (unpaired_msa_a3m, unpaired_msa_dir, etc.)

    Args:
        data_path: Directory containing input.json with msa_path references,
                   or directory with input.json + A3M files.
        device: GPU device number (default: 0).
        model_dir: Path to model parameters. If None, uses default.
        db_dir: Path to databases (needed for template search). If None, uses default.
        run_template_search: Whether to search for templates using MSA (default: True).
        max_template_date: Maximum template release date in YYYY-MM-DD format.
        unpaired_msa_a3m: Path to single unpaired A3M file (optional).
        paired_msa_a3m: Path to single paired A3M file (optional).
        unpaired_msa_dir: Directory containing {chain_id}.a3m files (optional).
        paired_msa_dir: Directory containing paired {chain_id}.a3m files (optional).
        run_inference: Whether to run model inference (default: True).
        flash_attention_implementation: 'triton', 'cudnn', or 'xla'.
        buckets: Token bucket sizes for compilation caching.

    Returns:
        PredictionResult object.

    Example:
        # With msa_path in input.json
        result = run_with_a3m("/path/to/data", run_template_search=True)

        # With command-line A3M file
        result = run_with_a3m(
            "/path/to/data",
            unpaired_msa_a3m="/path/to/protein.a3m",
        )

        # Without template search (faster)
        result = run_with_a3m("/path/to/data", run_template_search=False)
    """
    af3_path = _get_af3_path()
    defaults = _get_default_paths(af3_path)

    # Resolve all paths to absolute paths (important since we change cwd to af3_path)
    data_path = _resolve_path(data_path)
    model_dir = _resolve_path(model_dir) or defaults["model_dir"]
    db_dir = _resolve_path(db_dir) or defaults["db_dir"]
    unpaired_msa_a3m = _resolve_path(unpaired_msa_a3m)
    paired_msa_a3m = _resolve_path(paired_msa_a3m)
    unpaired_msa_dir = _resolve_path(unpaired_msa_dir)
    paired_msa_dir = _resolve_path(paired_msa_dir)

    # Find input.json
    if data_path.is_file() and data_path.name == "input.json":
        input_json = data_path
        output_dir = data_path.parent
    else:
        input_json = data_path / "input.json"
        output_dir = data_path

    if not input_json.exists():
        raise FileNotFoundError(f"input.json not found at {input_json}")

    job_name = output_dir.name
    logger.info(f"Running A3M mode for {job_name}")
    logger.info(f"Template search: {'enabled' if run_template_search else 'disabled'}")

    cmd = [
        sys.executable,
        str(af3_path / "run_alphafold_with_a3m.py"),
        f"--input_json={input_json}",
        f"--model_dir={model_dir}",
        f"--db_dir={db_dir}",
        f"--output_dir={output_dir}",
        f"--run_template_search={str(run_template_search).lower()}",
        f"--max_template_date={max_template_date}",
        f"--run_inference={str(run_inference).lower()}",
        f"--flash_attention_implementation={flash_attention_implementation}",
    ]

    if unpaired_msa_a3m:
        cmd.append(f"--unpaired_msa_a3m={unpaired_msa_a3m}")
    if paired_msa_a3m:
        cmd.append(f"--paired_msa_a3m={paired_msa_a3m}")
    if unpaired_msa_dir:
        cmd.append(f"--unpaired_msa_dir={unpaired_msa_dir}")
    if paired_msa_dir:
        cmd.append(f"--paired_msa_dir={paired_msa_dir}")
    if buckets:
        cmd.append(f"--buckets={','.join(map(str, buckets))}")

    result = _run_command(cmd, device=device, cwd=af3_path, job_name=job_name)
    result.output_dir = str(output_dir)
    return result


def run_with_msa(
    data_path: Union[str, Path],
    device: int = 0,
    model_dir: Optional[Union[str, Path]] = None,
    db_dir: Optional[Union[str, Path]] = None,
    flash_attention_implementation: str = "triton",
    buckets: Optional[list[int]] = None,
    convert_to_pdb: bool = False,
) -> PredictionResult:
    """Run AlphaFold3 with pre-computed MSA and templates (inference only).

    This is the fastest mode when you already have MSA and templates embedded
    in the input.json. It skips both MSA search and template search, running
    only the inference stage.

    The input.json must already contain:
    - unpairedMsa: The pre-computed MSA
    - pairedMsa: The paired MSA (can be empty string)
    - templates: List of template structures

    Args:
        data_path: Directory containing input.json with embedded MSA/templates.
        device: GPU device number (default: 0).
        model_dir: Path to model parameters. If None, uses default.
        db_dir: Path to databases. If None, uses default.
        flash_attention_implementation: 'triton', 'cudnn', or 'xla'.
        buckets: Token bucket sizes for compilation caching.
        convert_to_pdb: Whether to convert output CIF files to PDB format.

    Returns:
        PredictionResult object.

    Example:
        # Run inference only on pre-computed data
        result = run_with_msa("/path/to/data")

        # With PDB conversion
        result = run_with_msa("/path/to/data", convert_to_pdb=True)
    """
    af3_path = _get_af3_path()
    defaults = _get_default_paths(af3_path)

    # Resolve all paths to absolute paths (important since we change cwd to af3_path)
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
        raise FileNotFoundError(f"input.json not found at {input_json}")

    job_name = output_dir.name
    logger.info(f"Running MSA-only mode for {job_name} (inference only)")

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

    if buckets:
        cmd.append(f"--buckets={','.join(map(str, buckets))}")

    result = _run_command(cmd, device=device, cwd=af3_path, job_name=job_name)
    result.output_dir = str(output_dir)

    # Convert CIF to PDB if requested
    if convert_to_pdb and result.success:
        logger.info(f"Converting CIF files to PDB format...")
        _convert_cif_to_pdb(output_dir)
        logger.success(f"CIF to PDB conversion complete")

    return result


def run_batch(
    input_dir: Union[str, Path],
    device: int = 0,
    model_dir: Optional[Union[str, Path]] = None,
    db_dir: Optional[Union[str, Path]] = None,
    run_template_search: bool = True,
    max_template_date: str = "2021-09-30",
    skip_existing: bool = True,
    output_marker: str = "model_output.cif",
    input_json_name: str = "input.json",
    run_inference: bool = True,
    flash_attention_implementation: str = "triton",
    buckets: Optional[list[int]] = None,
) -> PredictionResult:
    """Run AlphaFold3 batch predictions on multiple variants.

    This is optimized for processing many variants efficiently:
    - Model is loaded once and reused for all predictions
    - Auto-detects input format (inline MSA/templates vs msa_path references)
    - Can skip already completed predictions

    Input format auto-detection:
    1. If input.json has inline unpairedMsa + templates: runs inference directly (fastest)
    2. If input.json has msa_path references: loads A3M, optionally searches templates
    3. If input.json has inline MSA only: optionally searches templates

    Args:
        input_dir: Directory containing variant subdirectories, each with input.json.
        device: GPU device number (default: 0).
        model_dir: Path to model parameters. If None, uses default.
        db_dir: Path to databases. If None, uses default.
        run_template_search: Whether to search templates (only used for msa_path inputs).
        max_template_date: Maximum template release date in YYYY-MM-DD format.
        skip_existing: Skip predictions if output files already exist (default: True).
        output_marker: File to check for determining if prediction is complete.
        input_json_name: Name of input JSON file in each directory (default: 'input.json').
        run_inference: Whether to run model inference (default: True).
        flash_attention_implementation: 'triton', 'cudnn', or 'xla'.
        buckets: Token bucket sizes for compilation caching.

    Returns:
        PredictionResult object with overall batch status.

    Example:
        # Process all variants in a directory
        result = run_batch("/path/to/variants", skip_existing=True)

        # Process without template search
        result = run_batch("/path/to/variants", run_template_search=False)

        # Use specific input file name
        result = run_batch("/path/to/variants", input_json_name="config.json")

    Directory structure expected:
        variants/
        ├── variant1/
        │   └── input.json
        ├── variant2/
        │   └── input.json
        └── variant3/
            └── input.json
    """
    af3_path = _get_af3_path()
    defaults = _get_default_paths(af3_path)

    # Resolve all paths to absolute paths (important since we change cwd to af3_path)
    input_dir = Path(input_dir).resolve()
    model_dir = _resolve_path(model_dir) or defaults["model_dir"]
    db_dir = _resolve_path(db_dir) or defaults["db_dir"]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    job_name = input_dir.name
    logger.info(f"Running batch mode for {job_name}")
    logger.info(f"Skip existing: {skip_existing}")
    logger.info(f"Template search: {'enabled' if run_template_search else 'disabled'}")

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
        f"--input_json_name={input_json_name}",
        f"--run_inference={str(run_inference).lower()}",
        f"--flash_attention_implementation={flash_attention_implementation}",
    ]

    if buckets:
        cmd.append(f"--buckets={','.join(map(str, buckets))}")

    result = _run_command(cmd, device=device, cwd=af3_path, job_name=job_name)
    result.output_dir = str(input_dir)
    return result


def _convert_cif_to_pdb(output_dir: Union[str, Path]) -> None:
    """Convert CIF files to PDB format using maxit tool."""
    import shutil

    if not shutil.which("maxit"):
        logger.warning("maxit tool not found. Skipping CIF to PDB conversion.")
        return

    output_dir = Path(output_dir)
    cif_files = list(output_dir.glob("*.cif"))

    for cif_file in cif_files:
        pdb_file = cif_file.with_suffix(".cif.pdb")
        subprocess.run(
            ["maxit", "-input", str(cif_file), "-output", str(pdb_file), "-o", "2"],
            capture_output=True,
        )
        logger.debug(f"Converted {cif_file.name} -> {pdb_file.name}")

    # Clean up maxit log
    maxit_log = output_dir / "maxit.log"
    if maxit_log.exists():
        maxit_log.unlink()


def set_log_level(level: str = "INFO") -> None:
    """Set the logging level for AF3 output.

    Args:
        level: Log level - DEBUG, INFO, WARNING, ERROR, or SUCCESS

    Example:
        set_log_level("DEBUG")  # Show all logs including debug
        set_log_level("WARNING")  # Only show warnings and errors
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>AF3</cyan> | {message}",
        level=level.upper(),
        colorize=True,
    )


def add_file_logger(log_file: Union[str, Path], level: str = "DEBUG") -> None:
    """Add a file logger for AF3 output.

    Args:
        log_file: Path to the log file
        level: Log level for the file

    Example:
        add_file_logger("af3_run.log")
    """
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | AF3 | {message}",
        level=level.upper(),
        rotation="10 MB",
    )
    logger.info(f"Added file logger: {log_file}")


# Convenience aliases
predict_structure = run_default
predict_with_msa = run_with_a3m
predict_msa_only = run_with_msa
predict_batch = run_batch


if __name__ == "__main__":
    # Simple CLI interface
    import argparse

    parser = argparse.ArgumentParser(
        description="AlphaFold3 structure prediction runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default full pipeline
  python alphafold3_runner.py default /path/to/data --device 0

  # With pre-computed A3M
  python alphafold3_runner.py a3m /path/to/data --run-template-search

  # MSA-only mode (fastest)
  python alphafold3_runner.py msa /path/to/data

  # Batch processing
  python alphafold3_runner.py batch /path/to/variants --skip-existing

  # With debug logging
  python alphafold3_runner.py default /path/to/data --log-level DEBUG
        """,
    )

    parser.add_argument(
        "mode",
        choices=["default", "a3m", "msa", "batch"],
        help="Prediction mode",
    )
    parser.add_argument(
        "data_path",
        help="Path to input data directory",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device number (default: 0)",
    )
    parser.add_argument(
        "--model-dir",
        help="Path to model parameters",
    )
    parser.add_argument(
        "--db-dir",
        help="Path to databases",
    )
    parser.add_argument(
        "--run-template-search",
        action="store_true",
        help="Run template search (for a3m mode)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing predictions (for batch mode)",
    )
    parser.add_argument(
        "--flash-attention",
        choices=["triton", "cudnn", "xla"],
        default="triton",
        help="Flash attention implementation (default: triton, use xla for non-A100)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        help="Optional log file path",
    )

    args = parser.parse_args()

    # Configure logging
    set_log_level(args.log_level)
    if args.log_file:
        add_file_logger(args.log_file)

    try:
        if args.mode == "default":
            results = run_default(
                args.data_path,
                device=args.device,
                model_dir=args.model_dir,
                db_dir=args.db_dir,
                flash_attention_implementation=args.flash_attention,
            )
            for r in results:
                status = "SUCCESS" if r.success else "FAILED"
                logger.info(f"{status}: {r.output_dir}")

        elif args.mode == "a3m":
            result = run_with_a3m(
                args.data_path,
                device=args.device,
                model_dir=args.model_dir,
                db_dir=args.db_dir,
                run_template_search=args.run_template_search,
                flash_attention_implementation=args.flash_attention,
            )
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"{status}: {result.output_dir}")

        elif args.mode == "msa":
            result = run_with_msa(
                args.data_path,
                device=args.device,
                model_dir=args.model_dir,
                db_dir=args.db_dir,
                flash_attention_implementation=args.flash_attention,
            )
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"{status}: {result.output_dir}")

        elif args.mode == "batch":
            result = run_batch(
                args.data_path,
                device=args.device,
                model_dir=args.model_dir,
                db_dir=args.db_dir,
                run_template_search=args.run_template_search,
                skip_existing=args.skip_existing,
                flash_attention_implementation=args.flash_attention,
            )
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"{status}: {result.output_dir}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
